#pragma once

#include <cmath>
#include <corecrt_math_defines.h>

namespace statmath {
	//https://stackoverflow.com/a/40260471
	static double erfinv(double x) {
		double tt1, tt2, lnx, sgn;
		sgn = 1 + -2 * signbit(x);

		x = (1 - x) * (1 + x);        // x = 1 - x*x;
		lnx = log(x);

		tt1 = 2 / (M_PI * 0.15449436008930206298828125) + 0.5 * lnx;
		tt2 = 1 / (0.15449436008930206298828125) * lnx;

		return(sgn * sqrt(-tt1 + sqrt(tt1 * tt1 - tt2)));
	}

	static double probit(double x) {
		return M_SQRT2 * erfinv(2 * x - 1);
	}

	static double gausspdf(double x, double variance) {
		static const double inv_sqrt_2pi = 0.3989422804014327;

		double v = x / variance;
		return inv_sqrt_2pi / variance * exp(-0.5f * v * v);
	}
}