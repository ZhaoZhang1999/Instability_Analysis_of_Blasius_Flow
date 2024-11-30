alpha = 0.179;						% parameter in O-S equation
Re = 1000;							% Reynolds number
dc = 5e-3;
[cr1, cr2] = deal(0, 1);			% range of real part of c
[ci1, ci2] = deal(-0.8, 0.2);		% range of imaginary part of c
[cr_grid_num, ci_grid_num] = deal(round((cr2 - cr1)/dc + 1), round((ci2 - ci1)/dc + 1));
grid_num = cr_grid_num*ci_grid_num;
p1_in_mats = cell(1, 2);			% real & imaginary parts of p1(0)

p1_in_r_mat = zeros(ci_grid_num, cr_grid_num);
p1_in_i_mat = zeros(ci_grid_num, cr_grid_num);
for R = 1:cr_grid_num
	for I = 1:ci_grid_num
		c = ((R - 1)*dc + cr1) + (1i)*((I - 1)*dc + ci1);
		[P, p1_in] = Runge_Kutta_OS(h, 2*dz, Re, U, Uii, alpha, c);
		[p1_in_r, p1_in_i] = deal(real(p1_in), imag(p1_in));
		p1_in_r_mat(I, R) = p1_in_r;
		p1_in_i_mat(I, R) = p1_in_i;
		progress = (R - 1)*ci_grid_num + I;
		progress = progress/grid_num*100;
		disp(['Re = ', num2str(Re, '%4d'), ': ', num2str(progress, '%4.3f'), '%']);
	end
end
p1_in_mats{1, 1} = p1_in_r_mat;
p1_in_mats{1, 2} = p1_in_i_mat;


function [P0, p1_in_0] = Runge_Kutta_OS(h, dz, Re, U, Uii, alpha, c)	% 4th-order Runge-Kutta method
	step_num = h/dz;
	beta = sqrt(alpha^2 + (1i)*alpha*Re*(1 - c));
	P0 = zeros(step_num + 1, 6);
	P0(1, :) = [1, -(alpha + beta), alpha^2 + alpha*beta + beta^2, alpha*beta, -alpha*beta*(alpha + beta), (alpha*beta)^2]*(alpha - beta)*exp(-(alpha + beta)*h);
	for n = 1:step_num
		[p1, p2, p3, p4, p5, p6] = deal(P0(n, 1), P0(n, 2), P0(n, 3), P0(n, 4), P0(n, 5), P0(n, 6));
		[s2, s4] = s2s4(alpha, Re, U, Uii, c, (step_num - n)*2 + 3);
		L1 = [p2, p3 + p4, s2*p2 + p5, p5, -s4*p1 + s2*p4 + p6, -s4*p2];
		[s2, s4] = s2s4(alpha, Re, U, Uii, c, (step_num - n)*2 + 2);
		L2 = [p2 - dz/2*L1(2), p3 - dz/2*L1(3) + p4 - dz/2*L1(4), s2*(p2 - dz/2*L1(2)) + p5 - dz/2*L1(5), p5 - dz/2*L1(5), -s4*(p1 - dz/2*L1(1)) + s2*(p4 - dz/2*L1(4)) + p6 - dz/2*L1(6), -s4*(p2 - dz/2*L1(2))];
		L3 = [p2 - dz/2*L2(2), p3 - dz/2*L2(3) + p4 - dz/2*L2(4), s2*(p2 - dz/2*L2(2)) + p5 - dz/2*L2(5), p5 - dz/2*L2(5), -s4*(p1 - dz/2*L2(1)) + s2*(p4 - dz/2*L2(4)) + p6 - dz/2*L2(6), -s4*(p2 - dz/2*L2(2))];
		[s2, s4] = s2s4(alpha, Re, U, Uii, c, (step_num - n)*2 + 1);
		L4 = [p2 - dz*L3(2), p3 - dz*L3(3) + p4 - dz*L3(4), s2*(p2 - dz*L3(2)) + p5 - dz*L3(5), p5 - dz*L3(5), -s4*(p1 - dz*L3(1)) + s2*(p4 - dz*L3(4)) + p6 - dz*L3(6), -s4*(p2 - dz*L3(2))];
		P0(n + 1, :) = P0(n, :) - dz/6*(L1 + 2*L2 + 2*L3 + L4);
	end
	p1_in_0 = P0(end, 1);
end

function [s2, s4] = s2s4(alpha, Re, U, Uii, c, m)
	s2 = 2*alpha^2 + (1i)*alpha*Re*(U(m) - c);
	s4 = -(alpha^4 + (1i)*alpha*Re*((U(m) - c)*alpha^2 + Uii(m)));
end