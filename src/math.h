#include <Eigen/Dense>

namespace nclr {
    template<typename T, int dim = 2>
    using Vector = Eigen::Matrix<T, dim, 1>;

    template<typename T, int dim = 2>
    using Matrix = Eigen::Matrix<T, dim, dim>;

    using real = float;

    inline auto diag(const float value) -> Matrix<real> {
        Matrix<real> m = Matrix<real>::Zero();
        m(0, 0) = value;
        m(1, 1) = value;
        return m;
    }


    inline auto nc_rand_int() -> uint32_t {
        static unsigned int x = 123456789, y = 362436069, z = 521288629, w = 88675123;
        unsigned int t = x ^ (x << 11);
        x = y;
        y = z;
        z = w;
        return (w = (w ^ (w >> 19)) ^ (t ^ (t >> 8)));
    }

    inline auto nc_rand() -> float { return nc_rand_int() * (1.0f / 4294967296.0f); }

    template<int dim>
    inline auto randvec() -> Vector<real, dim> {
        Vector<real, dim> ret;
#pragma unroll
        for (int i = 0; i < dim; ++i) { ret(i) = nc_rand(); }
        return ret;
    }
    template<int dim>
    inline auto constmat(const float value) -> Matrix<real, dim> {
        return Matrix<real, dim>::Constant(value);
    }

    template<int dim>
    inline auto constvec(const float value) -> Vector<real, dim> {
        return Vector<real, dim>::Constant(value);
    }

    template<int dim>
    inline auto nclr_svd(const Matrix<real, dim> &a, Matrix<real, dim> &U, Matrix<real, dim> &sig, Matrix<real, dim> &V)
            -> void {
        const auto svd = Eigen::JacobiSVD<Matrix<real>>(a, Eigen::ComputeFullU | Eigen::ComputeFullV);
        U = svd.matrixU();
        V = svd.matrixV();
        const auto values = svd.singularValues();
#pragma unroll
        for (int ii = 0; ii < dim; ++ii) { sig(ii, ii) = values(ii); }
    }

    template<int dim>
    inline auto nclr_polar(const Matrix<real, dim> &m, Matrix<real, dim> &R, Matrix<real, dim> &S) -> void {
        Matrix<real, dim> sig;
        Matrix<real, dim> U, V;
        nclr_svd(m, U, sig, V);

        R = U * V.transpose();
        S = V * sig * V.transpose();
    }
}// namespace nclr
