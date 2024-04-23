% ***************** SCALING ANALYSIS OF ASENIO COMBUSTION MODEL - NEW FORMULATION ************************************ %
% Author  : SIVA VIKNESH
% Email   : siva.viknesh@sci.utah.edu / sivaviknesh14@gmail.com
% Address : SCI INSTITUTE, UNIVERSITY OF UTAH, SALT LAKE CITY, UTAH, USA
% ******************************************************************************************************************** %
clear all;
clc
% PARAMETERS OF WILDFIRE PDE EQUATION
N      = 512;
Da     = logspace(-30, 30, N);
phi    = logspace(-30, 30, N);
[D, P] = meshgrid (Da, phi);
eps    = 3e-2;
beta   = 1.0;
q      = 1;
u_int  = 31.0;
k      = 1e-1;
alpha  = 1e-3;
upc    = 3.0;
w      = 1.0; 

% MAXIMUM TEMPERATURE OF FIREFRONT
umax   = q/eps *(beta + eps/q* u_int);

du     = umax - upc;
delta  = 9e-4;

K      = k*(1+eps*umax)^3 +1 ;
% HEATING TERM
HT = rdivide((K*du)/(delta^2) + (3*k*eps*(1+eps*umax)^2)*(du/delta)^2, D) + beta*exp(du/(1+eps*du)) ;

% COOLING TERM
CT = rdivide((w*du)/delta, P) + alpha*umax ;

% FLAME EVOLUTION
du_dt = HT - CT;

[c, h] = contour(D, P, du_dt, [0,0]); 
%set(gca, 'XScale', 'log');
%set(gca, 'YScale', 'log');
% % Adjust the axis limits 
loglog(c(1, :), c(2,:), 'LineWidth',2)
xlabel('Da')
ylabel('PHI') 
xlim([1e-10,1e10])
ylim([1e-10,1e10]);

%%
hold on
shg
pause(2)

w      = 1e-1; 

% MAXIMUM TEMPERATURE OF FIREFRONT
umax   = q/eps *(beta + eps/q* u_int);

du     = umax - upc;
delta  = 1e-5;

K      = k*(1+eps*umax)^3 +1 ;
% HEATING TERM
HT = rdivide((K*du)/(delta^2) + (3*k*eps*(1+eps*umax)^2)*(du/delta)^2, D) + beta*exp(du/(1+eps*du)) ;

% COOLING TERM
CT = rdivide((w*du)/delta, P) + alpha*umax ;

% FLAME EVOLUTION
du_dt = HT - CT;

[c, h] = contour(D, P, du_dt, [0,0]); 
%set(gca, 'XScale', 'log');
%set(gca, 'YScale', 'log');
% % Adjust the axis limits 
loglog(c(1, :), c(2,:), 'LineWidth',2)
xlabel('Da')
ylabel('PHI') 
xlim([1e-5,1e20])
ylim([1e-10,1e10]);

hold on
%%
filename = "Neutral_curve.dat";
fileID = fopen (filename,'w');
fprintf(fileID, 'variables = Da, Phi \n');
fprintf(fileID, '%6.14f %6.14f \n',c);
fclose(fileID);

%% EFFECT OF PDE STIFFNESS - epsilon
N      = 512;
Da     = logspace(-30, 30, N);
phi    = logspace(-30, 30, N);
[D, P] = meshgrid (Da, phi);

Ne = 5;
epsilon = logspace(-1, 3, Ne);

for i = 1: Ne
    % MAXIMUM TEMPERATURE OF FIREFRONT
    epsilon(i)
    umax   = q/epsilon(i) *(beta + epsilon(i)/q* u_int);

    du     = umax - upc;
    delta  = 1e-3;

    K      = k*(1+epsilon(i)*umax)^3 +1 ;
    % HEATING TERM
    HT = rdivide((K*du)/(delta^2) + (3*k*epsilon(i)*(1+epsilon(i)*umax)^2)*(du/delta)^2, D) + beta*exp(du/(1+epsilon(i)*du)) ;

    % COOLING TERM
    CT = rdivide((w*du)/delta, P) + alpha*umax ;

    % FLAME EVOLUTION
    du_dt = HT - CT;

    [c, h] = contour(D, P, du_dt, [0,0]); 
    filename =  strcat("Effect_of_epsilon_", num2str(epsilon(i)), ".dat");
    fileID = fopen (filename,'w');
    fprintf(fileID, 'variables = Da, Phi \n');
    fprintf(fileID, '%6.14f %6.14f \n',c);
    fclose(fileID);
  
end

%% EFFECT OF THERMAL CONDUCTIVITY
Nk = 5;
k = logspace(-2, 2, Nk);

for i = 1: Nk

    K      = k(i)*(1+eps*umax)^3 +1 ;
    % HEATING TERM
    HT = rdivide((K*du)/(delta^2) + (3*k(i)*eps*(1+eps*umax)^2)*(du/delta)^2, D) + beta*exp(du/(1+eps*du)) ;

    % COOLING TERM
    CT = rdivide((w*du)/delta, P) + alpha*umax ;

    % FLAME EVOLUTION
    du_dt = HT - CT;

    [c, h] = contour(D, P, du_dt, [0,0]); 
    filename =  strcat("Effect_of_thermal_cond_", num2str(k(i)), ".dat");
    fileID = fopen (filename,'w');
    fprintf(fileID, 'variables = Da, Phi \n');
    fprintf(fileID, '%6.14f %6.14f \n',c);
    fclose(fileID);

end

%% EFFECT OF PHASE CHANGE TEMPERATURE

% PARAMETERS OF WILDFIRE PDE EQUATION
N      = 512;
Da     = logspace(-10, 10, N);
phi    = logspace(-10, 10, N);
[D, P] = meshgrid (Da, phi);
eps    = 3e-2;
beta   = 1.0;
q      = 1;
u_int  = 31.0;
k      = 1e-1;
alpha  = 1e-3;

w      = 1.0; 

% MAXIMUM TEMPERATURE OF FIREFRONT
umax   = q/eps *(beta + eps/q* u_int);
Nu     = 5;
upc    = linspace (0.0, 1.0, Nu)*umax;

for i = 1: Nu
    du     = umax - upc(i);
    delta  = 1e-5;

    K      = k*(1+eps*umax)^3 +1 ;
    % HEATING TERM
    HT = rdivide((K*du)/(delta^2) + (3*k*eps*(1+eps*umax)^2)*(du/delta)^2, D) + beta*exp(du/(1+eps*du)) ;

    % COOLING TERM
    CT = rdivide((w*du)/delta, P) + alpha*umax ;

    % FLAME EVOLUTION
    du_dt = HT - CT;

    [c, h] = contour(D, P, du_dt, [0,0]);
    filename =  strcat("Effect_of_phase_chang_tempera", num2str(upc(i)), ".dat");
    fileID = fopen (filename,'w');
    fprintf(fileID, 'variables = Da, Phi \n');
    fprintf(fileID, '%6.14f %6.14f \n',c);
    fclose(fileID);

end
