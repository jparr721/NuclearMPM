#include <Eigen/Dense>
#include <iostream>

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

    template<int res>
    inline auto cube(real xmin, real xmax, real ymin, real ymax) -> std::vector<Vector<real, 2>> {
        Vector<real, res> x = Vector<real, res>::LinSpaced(xmin, xmax);
        Vector<real, res> y = Vector<real, res>::LinSpaced(ymin, ymax);

        std::vector<Vector<real, 2>> all_pts;
        for (auto r = 0; r < x.rows(); ++r) {
            for (auto c = 0; c < y.rows(); ++c) { all_pts.emplace_back(x(r), y(c)); }
        }
        return all_pts;
    }

    template<int res>
    inline auto gyroid(real k, real t, const Vector<real, 2> &pos) -> real {
        const auto two_pi = (2.0 * M_PI) / k;
        const auto x = pos(0);
        const auto y = pos(1);
        const auto z = 0.0;
        /* const auto z = pos(2); */

        /* return std::sin(two_pi * x) * std::cos(two_pi * y); */
        return std::sin(two_pi * x) * std::cos(two_pi * y) + std::sin(two_pi * y) * std::cos(two_pi * z) +
               std::sin(two_pi * z) * std::cos(two_pi * x) - t;
    }

    /**
     * igl::grid function
     */
    template<typename Derivedres, typename DerivedGV>
    inline auto grid(const Eigen::MatrixBase<Derivedres> &res, Eigen::PlainObjectBase<DerivedGV> &GV) -> void {
        using namespace Eigen;
        typedef typename DerivedGV::Scalar Scalar;
        GV.resize(res.array().prod(), res.size());
        const auto lerp = [&res](const Scalar di, const int d) -> Scalar { return di / (Scalar) (res(d) - 1); };
        int gi = 0;
        Derivedres sub;
        sub.resizeLike(res);
        sub.setConstant(0);
        for (int gi = 0; gi < GV.rows(); gi++) {
            // omg, I'm implementing addition...
            for (int c = 0; c < res.size() - 1; c++) {
                if (sub(c) >= res(c)) {
                    sub(c) = 0;
                    // roll over
                    sub(c + 1)++;
                }
            }
            for (int c = 0; c < res.size(); c++) { GV(gi, c) = lerp(sub(c), c); }
            sub(0)++;
        }
    }

    /* auto cube(int xmin, int xmax, int ymin, int ymax, int zmin, int zmax, int res) -> std::vector<Vector<real, 3>> {} */

}// namespace nclr
