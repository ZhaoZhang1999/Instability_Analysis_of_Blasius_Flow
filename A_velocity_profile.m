clear; close all; clc;

h = 10;					% computational domain
dz = 0.005;				% space step
tol_of_fi_out = 1e-8;	% tolerrance
fi_out = 1;
fii_in_1 = 0;			% initial value of f''(0)
fii_in_2 = 1;			% initial value of f''(0)
while true
	[F1, fi_out_1] = Runge_Kutta_Blasius(h, dz, fii_in_1);
	[F2, fi_out_2] = Runge_Kutta_Blasius(h, dz, fii_in_2);
	[distance1, distance2] = deal(fi_out_1 - fi_out, fi_out_2 - fi_out);
	if abs(distance2) < tol_of_fi_out
		break;
	end
	fii_in_3 = fii_in_2 - distance2*(fii_in_2 - fii_in_1)/(distance2 - distance1);		% secant method
	[fii_in_1, fii_in_2] = deal(fii_in_2, fii_in_3);
end
Z = 0:dz:h;
U = F2(:, 2);						% velocity profile
Uii = -1/2*F2(:, 1).*F2(:, 3);		% U''
disp(['f"(0) = ', num2str(fii_in_2, '%.8f')]);

function [F0, fi_out_0] = Runge_Kutta_Blasius(h, dz, fii_in_0)		% 4th-order Runge-Kutta method
	step_num = h/dz;
	F0 = zeros(step_num + 1, 3);
	F0(1, 3) = fii_in_0;
	for n = 1:step_num
		[f, fi, fii] = deal(F0(n, 1), F0(n, 2), F0(n, 3));
		K1 = [fi, fii, -1/2*f*fii];
		K2 = [fi + dz/2*K1(2), fii + dz/2*K1(3), -1/2*(f + dz/2*K1(1))*(fii + dz/2*K1(3))];
		K3 = [fi + dz/2*K2(2), fii + dz/2*K2(3), -1/2*(f + dz/2*K2(1))*(fii + dz/2*K2(3))];
		K4 = [fi + dz*K3(2), fii + dz*K3(3), -1/2*(f + dz*K3(1))*(fii + dz*K3(3))];
		F0(n + 1, :) = F0(n, :) + dz/6*(K1 + 2*K2 + 2*K3 + K4);
	end
	fi_out_0 = F0(end, 2);
end