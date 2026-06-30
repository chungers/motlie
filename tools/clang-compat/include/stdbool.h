#ifndef MOTLIE_CLANG_COMPAT_STDBOOL_H
#define MOTLIE_CLANG_COMPAT_STDBOOL_H

#ifndef __cplusplus
#define bool _Bool
#define true 1
#define false 0
#else
#define _Bool bool
#endif

#define __bool_true_false_are_defined 1

#endif
