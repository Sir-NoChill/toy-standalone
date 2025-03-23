#ifndef __TOY_MACROS_
#define __TOY_MACROS_

#include <cstdio>
#include <cstdarg>
#include <cassert>

/**
 * Prints the function information for a function call when
 * DEBUG is defined.
 */
#ifdef DEBUG
    #define DEBUG_PRINT() std::cerr << "-- CALL: " << __FUNCTION__ << std::endl;
#else
    #define DEBUG_PRINT()
#endif

/**
 * Prints a debug message if DEBUG is defined.
 *
 * Call similar to printf(...)
 */
#ifdef DEBUG
    #define DEBUG_PRINTF(fmt, ...) \
        do { \
            fprintf(stderr, fmt, __VA_ARGS__); \
        } while (0)
#else
    #define DEBUG_PRINTF(fmt, ...) \
        do { } while (0)
#endif

/**
 * Marks an assertion for debug execution if DEBUG is defined.
 */
#ifdef DEBUG
    #define DEBUG_ASSERT(cond, msg) \
        do { \
            if (!(cond)) { \
                std::cerr << "Assertion failed: (" #cond "), function " << __FUNCTION__ << ", file " << __FILE__ << ", line " << __LINE__ << ": " << msg << std::endl; \
                assert(cond); \
            } \
        } while (0)
#else
    #define DEBUG_ASSERT(cond, msg) \
        do { } while (0)
#endif

#endif
