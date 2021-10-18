function dX_dtau = ascent_odes_tf(tau,X,tf)
    %
    % State and Costate Differential Equation Function for the Flat-Earth
    % Optimal Ascent Problem with Atmospheric Drag and linearly time-varying
    % mass
    %
    %
    % The independent variable here is the nondimensional time, tau, the state
    % vector is X, and the final time, tf, is an unknown parameter that must
    % also be passed to the DE function.
    % Note that the state vector X has components
    % X(1) = xbar, horizontal component of position
    % X(2) = ybar, vertical component of position
    % X(3) = Vxbar, horizontal component of velocity
    % X(4) = Vybar, vertical component of velocity
    % X(5) = lambda_2_bar, first costate
    % X(6) = lambda_3_bar, second costate
    % X(7) = lambda_4_bar, third costate
    % pass in values of relevant constants as global variables
    global g_accel Vc h eta beta f m0 mdot
    m = m0-abs(mdot)*tau*tf;
    % State and Costate differential equations in terms of d/dt:
    xbardot = X(3)*Vc/h;
    ybardot = X(4)*Vc/h;
    Vxbardot = (f/Vc*(-X(6)/sqrt(X(6)^2+X(7)^2)) ...
        -eta*exp(-X(2)*beta)*X(3)*sqrt(X(3)^2+X(4)^2)*Vc)/m;
    Vybardot = (f/Vc*(-X(7)/sqrt(X(6)^2+X(7)^2)) ...
        -eta*exp(-X(2)*beta)*X(4)*sqrt(X(3)^2+X(4)^2)*Vc)/m-g_accel/Vc;
    if sqrt(X(3)^2+X(4)^2) == 0
        lambda_2_bar = 0;
        lambda_3_bar = 0;
        lambda_4_bar = -X(5)*Vc/h;
    else
        lambda_2_bar = ...
            -(X(6)*X(3)+X(7)*X(4))*eta*beta*exp(-X(2)*beta)*sqrt(X(3)^2+X(4)^2)*Vc/m;
        lambda_3_bar = eta*exp(-X(2)*beta)*Vc*(X(6)*(2*X(3)^2+X(4)^2) ...
            + X(7)*X(3)*X(4))/sqrt(X(3)^2+X(4)^2)/m;
        lambda_4_bar = -X(5)*Vc/h+eta*exp(-X(2)*beta)*Vc*(X(7)*(X(3)^2 ...
            +2*X(4)^2) ...
            +X(6)*X(3)*X(4))/sqrt(X(3)^2+X(4)^2)/m;
    end
    % Nondimensionalize time (with tau = t/tf and d/dtau = tf*d/dt). We must
    % multiply each differential equation by tf to convert our derivatives from
    % d/dt to d/dtau.
    dX_dtau = tf*[xbardot; ybardot; Vxbardot; Vybardot; ...
        lambda_2_bar; lambda_3_bar; lambda_4_bar];
    return