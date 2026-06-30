#ifndef MOTLIE_CLANG_COMPAT_STDDEF_H
#define MOTLIE_CLANG_COMPAT_STDDEF_H

#ifndef __cplusplus
#define NULL ((void *)0)
#else
#define NULL 0
#endif

typedef __PTRDIFF_TYPE__ ptrdiff_t;
typedef __SIZE_TYPE__ size_t;
typedef __WCHAR_TYPE__ wchar_t;

#if defined(__need_ptrdiff_t) || defined(__need_size_t) || defined(__need_wchar_t) || defined(__need_NULL)
#undef __need_ptrdiff_t
#undef __need_size_t
#undef __need_wchar_t
#undef __need_NULL
#endif

#endif
