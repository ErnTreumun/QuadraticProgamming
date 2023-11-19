function [x, objval] = FWOptStep(Q, c, Adesg, bdesg, Aeq, beq, xmin, xmax, max_iter, X0,Opt)
    % Minimizando una función de la forma
    % x^T Q x + c^T x sujeto a
    % Adesg x <= bdesg; Aeq x = beq; xmín <= x <= xmáx
    % xmin, Aquí Opt permite utilizar o no un paso óptimo
    %Con opt= 0 NO hacemos paso óptimo y con Opt = 1 sí
    % <= x <=xmax partiendo de X0 y con un máximo de iteracioens max_iter
    % Inicialización
    [~, n] = size(Q);
    x = X0;

    % Opciones para suprimir mensajes de salida de linprog
    options = optimoptions('linprog', 'Display', 'off');
    %options = optimoptions('fmincon', 'Algorithm', 'interior-point', 'MaxIterations', 1, 'Display', 'off');
    %options = optimoptions('quadprog', 'MaxIterations',1, 'Display', 'off');
    
    for iter = 1:max_iter
        % Calcular el gradiente
        gradient = (Q' +Q) * x + c;
        
        % Resolver subproblema de minimización restringida (búsqueda lineal)
        direction= linprog(gradient,Adesg,bdesg,Aeq,beq,xmin,xmax,options);
        %[direction, ~] = fmincon(@(x)quadratic(x,A,b),x,[],[],Aeq,beq,xmin,xmax,[], options);
        %[direction,~]= quadprog([],gradient,[],[],Aeq,beq,xmin,xmax,x,options);
        
        if Opt==0
            % Calcular el tamaño del paso (paso óptimo)
            step_size = 2 / (iter + 2);

        else
            A1 = x' * Q * x;
            B1 = x'*(Q + Q')*direction;
            C1 = direction'*Q*direction;
            alpha = A1 - B1 + C1;
            beta = B1 - 2*A1 + c'*(direction - x);
            func = @(z) alpha* z^2 + beta*z;
            % Configurar las opciones de visualización para fminbnd
            options = optimset('Display', 'off');
            step_size= fminbnd(func,0,1,options);
            %Calcular explícito
        end

        % Actualizar la solución
        x = (1- step_size)*x + step_size * direction;
        
        % Calcular el valor objetivo y la norma del gradiente
        objval = x'*Q*x + c'*x;
    end
end
