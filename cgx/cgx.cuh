#pragma once

#include <iostream>
#include <cstdint>
#include <assert.h>

#define GLOBALQUALIFIER __global__
#define HOSTDEVICEQUALIFIER __host__ __device__

// safe division
#define SDIV(x,y)(((x)+(y)-1)/(y))

#if CUDART_VERSION >= 9000

namespace cgx {

    HOSTDEVICEQUALIFIER
    void hold_and_catch_fire () {
        assert(0); //mharris style
    }

    class dim3_t {

    public:

        int64_t x, y, z;

        HOSTDEVICEQUALIFIER
        dim3_t () : x(1), y(1), z(1) {}

        HOSTDEVICEQUALIFIER
        dim3_t (int64_t x_, int64_t y_, int64_t z_) : x(x_), y(y_), z(z_) {}
    };


    class range {

        const uint64_t stride, offset;
        const dim3_t lower, upper, steps, dims;

        class range_iterator {
            uint64_t state;
            const uint64_t stride;
            const dim3_t lower, upper, steps, dims;

        public:

            constexpr static uint64_t null_state=~uint64_t(0);

            HOSTDEVICEQUALIFIER
            range_iterator(
                const dim3_t& lower_,
                const dim3_t& upper_,
                const dim3_t& steps_,
                const dim3_t& dims_,
                const uint64_t state_,
                const uint64_t stride_) :
                    lower  (lower_),
                    upper  (upper_),
                    steps  (steps_),
                    state  (state_),
                    stride (stride_),
                    dims   (dims_) {}

            HOSTDEVICEQUALIFIER
            dim3_t operator* () const {

                dim3_t result;
                uint64_t state_ = state;

                result.x = (state_ % dims.x)*steps.x+lower.x;
                state_ /= dims.x;
                result.y = (state_ % dims.y)*steps.y+lower.y;
                state_ /= dims.y;
                result.z = (state_ % dims.z)*steps.z+lower.z;

                return result;
            }

            HOSTDEVICEQUALIFIER
            const range_iterator& operator++ () {

                state = state+stride < uint64_t(dims.x*dims.y*dims.z)
                      ? state+stride : null_state;

                return *this;
            }

            HOSTDEVICEQUALIFIER
            bool operator!=(
                const range_iterator& other) const {

                return state != other.state;
            }
        };

    public:

        template <typename cg_t> HOSTDEVICEQUALIFIER
        range(
            const cg_t& cg,
            const dim3_t& lower_,
            const dim3_t& upper_,
            const dim3_t& steps_=dim3_t(1, 1, 1)) :
                lower  (lower_)           ,
                upper  (upper_)           ,
                steps  (steps_)           ,
                offset (cg.thread_rank()) ,
                stride (cg.size())        ,
                dims   (dim3_t(SDIV(upper_.x-lower.x, steps_.x),
                               SDIV(upper_.y-lower.y, steps_.y),
                               SDIV(upper_.z-lower.z, steps_.z))) {

            validate();
        }

        template <typename cg_t> HOSTDEVICEQUALIFIER
        range(
            const cg_t& cg,
            const dim3_t& upper_) :
                lower  (dim3_t(0, 0, 0))  ,
                upper  (upper_         )  ,
                steps  (dim3_t(1, 1, 1))  ,
                offset (cg.thread_rank()) ,
                stride (cg.size())        ,
                dims   (dim3_t(upper_.x, upper_.y, upper_.z)) {

            validate();
        }

        template <typename cg_t> HOSTDEVICEQUALIFIER
        range(
            const cg_t& cg,
            const int64_t& lower_,
            const int64_t& upper_,
            const int64_t& steps_=1) :
                lower  (dim3_t(lower_, 0, 0)) ,
                upper  (dim3_t(upper_, 1, 1)) ,
                steps  (dim3_t(steps_, 1, 1)) ,
                offset (cg.thread_rank())     ,
                stride (cg.size())            ,
                dims   (dim3_t(SDIV(upper_-lower_, steps_), 1, 1)) {

            validate();
        }

        template <typename cg_t> HOSTDEVICEQUALIFIER
        range(
            const cg_t& cg,
            const int64_t& upper_) :
                lower  (dim3_t(0     , 0, 0)) ,
                upper  (dim3_t(upper_, 1, 1)) ,
                steps  (dim3_t(1     , 1, 1)) ,
                offset (cg.thread_rank())     ,
                stride (cg.size())            ,
                dims   (dim3_t(upper_, 1, 1)) {

            validate();
        }

        HOSTDEVICEQUALIFIER
        void validate () {
            if (steps.x <= 0 || steps.y <= 0 || steps.z <= 0)
                hold_and_catch_fire();
        }

        HOSTDEVICEQUALIFIER
        range_iterator begin () {
            return offset >= uint64_t(dims.x*dims.y*dims.z) ? end() :
                   range_iterator(lower, upper, steps, dims, offset, stride);
        }

        HOSTDEVICEQUALIFIER
        range_iterator end () {
            return range_iterator(lower, upper, steps, dims,
                                  range_iterator::null_state, stride);
        }
    };
}

#else

#pragma message("Currently only working with CUDA 9 or above.")

#endif
