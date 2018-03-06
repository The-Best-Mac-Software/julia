// This file is a part of Julia. License is MIT: https://julialang.org/license

#include "llvm-version.h"
#include <map>
#include <string>
#include <cstdio>
#include <llvm/Support/Host.h>
#include "julia.h"
#include "julia_internal.h"
#include "processor.h"
#include "julia_assert.h"

using namespace llvm;

// --- library symbol lookup ---

// map from user-specified lib names to handles
static std::map<std::string, void*> libMap;
static jl_mutex_t libmap_lock;
extern "C"
void *jl_get_library(const char *f_lib)
{
    void *hnd;
#ifdef _OS_WINDOWS_
    if (f_lib == JL_EXE_LIBNAME)
        return jl_exe_handle;
    if (f_lib == JL_DL_LIBNAME)
        return jl_dl_handle;
#endif
    if (f_lib == NULL)
        return jl_RTLD_DEFAULT_handle;
    JL_LOCK_NOGC(&libmap_lock);
    // This is the only operation we do on the map, which doesn't invalidate
    // any references or iterators.
    void **map_slot = &libMap[f_lib];
    JL_UNLOCK_NOGC(&libmap_lock);
    hnd = jl_atomic_load_acquire(map_slot);
    if (hnd != NULL)
        return hnd;
    // We might run this concurrently on two threads but it doesn't matter.
    hnd = jl_load_dynamic_library(f_lib, JL_RTLD_DEFAULT);
    if (hnd != NULL)
        jl_atomic_store_release(map_slot, hnd);
    return hnd;
}

extern "C" JL_DLLEXPORT
void *jl_load_and_lookup(const char *f_lib, const char *f_name, void **hnd)
{
    void *handle = jl_atomic_load_acquire(hnd);
    if (!handle)
        jl_atomic_store_release(hnd, (handle = jl_get_library(f_lib)));
    return jl_dlsym(handle, f_name);
}

// miscellany
std::string jl_get_cpu_name_llvm(void)
{
    return llvm::sys::getHostCPUName().str();
}

std::string jl_get_cpu_features_llvm(void)
{
    StringMap<bool> HostFeatures;
    llvm::sys::getHostCPUFeatures(HostFeatures);
    std::string attr;
    for (auto &ele: HostFeatures) {
        if (ele.getValue()) {
            if (!attr.empty()) {
                attr.append(",+");
            }
            else {
                attr.append("+");
            }
            attr.append(ele.getKey().str());
        }
    }
    // Explicitly disabled features need to be added at the end so that
    // they are not reenabled by other features that implies them by default.
    for (auto &ele: HostFeatures) {
        if (!ele.getValue()) {
            if (!attr.empty()) {
                attr.append(",-");
            }
            else {
                attr.append("-");
            }
            attr.append(ele.getKey().str());
        }
    }
    return attr;
}

extern "C" JL_DLLEXPORT
jl_value_t *jl_get_JIT(void)
{
    const std::string& HostJITName = "ORCJIT";
    return jl_pchar_to_string(HostJITName.data(), HostJITName.size());
}


static htable_t trampolines;

static void trampoline_deleter(jl_value_t *o)
{
    void **nvals = (void**)ptrhash_get(&trampolines, (void*)o);
    ptrhash_remove(&trampolines, (void*)o);
    assert(nvals && nvals != HT_NOTFOUND);
    free(nvals[0]); // TODO: return to RWX Pool
    free(nvals);
}

extern "C" JL_DLLEXPORT
void *jl_get_cfunction_trampoline(
    htable_t *cache, // weakref htable indexed by (f, vals)
    jl_value_t *finalizer, // cleanup when this is deleted
    void *(*init_trampoline)(void *tramp, void **nval),
    jl_value_t *f,
    jl_svec_t *fill,
    jl_unionall_t *env,
    jl_value_t **vals)
{
    // lookup (f, vals) in cache
    if (!cache->table)
        htable_new(cache, 1);
    if (fill != jl_emptysvec) {
        htable_t **cache2 = (htable_t**)ptrhash_bp(cache, (void*)vals);
        cache = *cache2;
        if (cache == HT_NOTFOUND) {
            cache = htable_new((htable_t*)malloc(sizeof(htable_t)), 1);
            *cache2 = cache;
        }
    }
    void *tramp = ptrhash_get(cache, (void*)f);
    if (tramp)
        return tramp;

    // not found, allocate a new one
    size_t n = jl_svec_len(fill);
    void **nval = (void**)malloc(sizeof(void**) * (n + 2));
    JL_TRY{
        nval[1] = (void*)f;
        for (size_t i = 0; i < n; i++) {
            jl_value_t *sparam_val = jl_instantiate_type_in_env(jl_svecref(fill, i), env, vals);
            if (sparam_val != (jl_value_t*)jl_any_type)
                if (!jl_is_concrete_type(sparam_val) || !jl_is_immutable(sparam_val))
                    sparam_val = NULL;
            nval[i + 2] = (void*)sparam_val;
        }
    }
    JL_CATCH {
        free(nval);
        jl_rethrow();
    }
    tramp = malloc(64); // TODO: use RWX pool
    nval[0] = tramp;
    tramp = init_trampoline(tramp, nval + 1);
    ptrhash_put(cache, (void*)f, tramp);
    int permanent =
        jl_is_concrete_type(finalizer) ||
        (((jl_datatype_t*)jl_typeof(finalizer))->instance == finalizer);
    if (jl_is_unionall(finalizer)) {
        jl_value_t *uw = jl_unwrap_unionall(finalizer);
        if (jl_is_datatype(uw) && ((jl_datatype_t*)uw)->name->wrapper == finalizer)
            permanent = true;
    }
    if (!permanent) {
        if (!trampolines.table)
            htable_new(&trampolines, 1);
        ptrhash_put(&trampolines, (void*)finalizer, (void*)nval);
        void *ptr_finalizer[2] = {
                (void*)jl_voidpointer_type,
                (void*)&trampoline_deleter
            };
        jl_gc_add_finalizer(finalizer, (jl_value_t*)&ptr_finalizer[1]);
    }
    return tramp;
}
