function [u,x] = FD_main(tend, N, mu, index)
% solving viscous burger's eqn
% u_t + (u^2/2)_x =\mu*u_xx
% by simple finite volume method


%N_array=[10 20 40 80 160];

% for j =5
% setup geometry
%  N =40; % x dimension
%  tend=1;
%  mu=0.001;
CFL =0.8;

xstart =-1;
xend = 1;%3;
%  tend = 1;%2;

% visous term
% mu = 0.001;

dx = (xend-xstart)/N;
x = (xstart:dx:xend)';
%dt/dx + 2*mu*dt/dx<1
dt = 1/2000;% 0.005;%CFL*dx/(2*pi)*0.1;%r*dx^2/mu; %;0.005;
nt = floor(tend/dt);



% coefficient
b = dt*mu/dx^2;
a = dt/dx;


%preallocate matrices
u = init(x,0,mu);
% plot(x,u);

% the linear part
e = ones(N+1,1);
A = spdiags([e -2*e e], -1:1, N+1, N+1);
%A = A(2:end-1,:);
A([1 end],:) = 0;


U_snapshot = [];
U_snapshot= [U_snapshot u];
F_snapshot = [];


for i = 1:nt
    % update nonlinear flux
    % upwind
    %F = flux(u(2:end)) - flux(u(1:end-1));

    % godnuov scheme
    us = rpbu(u(1:end-1), u(2:end));
    F = flux(us(2:end))-flux(us(1:end-1))  ;
    F = [0; F; 0]; 
    u = u - a*F + b*A*u  ; 
    
    t= i*dt;
    % update boundary value
    u(1) = 0;%init(xstart,t,mu);
    u(end) = 0;%init(xend,t,mu);
    
    % store all the snapshots
    U_snapshot = [U_snapshot u];
    F_snapshot = [F_snapshot F];
    
%     if mod(i,50)==1
%         set(0,'defaultlinelinewidth',2)
%         plot(x,u,'-o'); hold on
%         drawnow;
%     end
end

% disp('check snapshots_burgerShock.mat LinearMatr_burgerShock.mat  in the local folder......')
% disp('---------------------------------------------')

%save('snapshots_burgerShock.mat', 'U_snapshot');
u = U_snapshot;
save(strcat('solutions/', num2str(index), '.mat'), 'u');
if(max(max(u) == Inf))
    disp(mu*50*pi);
end
if(min(min(u)) == -Inf)
    disp(mu*50*pi);
end
end

function y = flux(u)
   % upwind, need to assume u=0
   y = 0.5*(u.^2);
   % average flux
   %y = 0.5*(u(2:end).^2 + u(1:end-1).^2);
end
function ret = rpbu( uL, uR )
usonic = 0.0;        % sonic point
s = 0.5 * (uL + uR); % shock speed from Rankine-Hugoniot
issonic = (uL < 0) & (0 < uR);
ret = issonic * usonic + (~issonic) .* (uL .* (s >= 0) + uR .* (s < 0));
end
function y = init(x,t,mu)
    %y = exp(-2*(x-1).^2);
    %y =1/4 + sin(2*pi*(2*x-1))/2;
    y = -sin(pi*x) ;
    %y = sin(x-2*pi*t);
end
