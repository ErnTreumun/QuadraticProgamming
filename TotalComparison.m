C=7e-3;
%Ts=1e-4;
Ts=1;
A=[1 0 0 -1 0 0; 1 0 0 0 -1 0; 1 0 0 0 0 -1; 0 1 0 -1 0 0; 0 1 0 0 -1 0; 0 1 0 0 0 -1; 0 0 1 -1 0 0; 0 0 1 0 -1 0; 0 0 1 0 0 -1;]';
Ecref=C*500^2*ones(9,1)/2;
E0=zeros(9,1);
lambda=0.3;
Q= eye(9);


Np=10;
I1=10;
omega1= 300;
omega2=110;
V1=300;
V2=110;

vchi = zeros(6*(Np-1),1);
v0 = zeros(9*(Np-1),1);
ichi=zeros(6*(Np-1),1);

t=0;
for i = 1:Np-1
    vchi((i-1)*6+1:i*6,1) = [V1*cos(omega1*t); V1*cos(omega1*t - 2*pi/3); V1*cos(omega1*t + 2*pi/3); V2*cos(omega2*t); V2*cos(omega2*t - 2*pi/3); V2*cos(omega2*t + 2*pi/3)];
    v0((i-1)*9+1:i*9,1) = [0;0;-1;1;2;3;0;2;3];
    ichi=[I1*cos(omega1*t); I1*cos(omega1*t - 2*pi/3); I1*cos(omega1*t + 2*pi/3); (I1/V1)*V2*cos(omega2*t); (I1/V1)*V2*cos(omega2*t - 2*pi/3); (I1/V1)*V2*cos(omega2*t + 2*pi/3)];
    t= t + i*Ts;
end

ichi=ones(9*(Np-1),1);

Aeq = [];
for i=1:(Np-1)
    Aeq= cat(2,Aeq,A);
end

beq= zeros(6,1);



[H1,b1]=RealModel(E0,Ecref,Ts,lambda,Np,v0,vchi,ichi,A,Q);

N1 = [1 1 1 1; -1 0 -1 0; 0 -1 0 -1; -1 -1 0 0; 1 0 0 0; 0 1 0 0; 0 0 -1 -1; 0 0 1 0; 0 0 0 1];

%Creando la matriz N diagonal por bloques
N = zeros([9, (Np-1)*4]);

for i=1:Np-1
    N((i-1)*size(N1,1)+1:i*size(N1,1), (i-1)*size(N1,2)+1:i*size(N1,2)) = N1;
end

%Matrices transformadas
H = (N')*H1*N;
b = (N')*b1;

[~,n2]=size(H1);
[~,n]= size(H);
X0 = zeros(n,1);
X01=zeros(n2,1);
imin=-5*ones(n2,1);
imax=5*ones(n2,1);

Bdesg = [-N;N];
bdesg= [-imin;imax];




%Inf es el mínimo cantidad de iteraciones que queremos que hagan y sup es
%el máximo
inf = 1;
sup = 50;
m= sup - inf + 1;

%Función nueva
fvalFWNSNew=zeros(m,1);
fvalFWSNew=zeros(m,1);
fvalIPNew=zeros(m,1);
fvalASNew=zeros(m,1);
fvalQPNew=zeros(m,1);


tiempoFWNSNew=zeros(m,1);
tiempoFWSNew=zeros(m,1);
tiempoIPNew=zeros(m,1);
tiempoASNew=zeros(m,1);
tiempoQPNew=zeros(m,1);


%Función antigua

fvalFrankWolfeNonStep=zeros(m,1);
fvalFrankWolfeStep=zeros(m,1);
fvalInteriorPoint=zeros(m,1);
fvalActSet=zeros(m,1);
fvalQuadProg=zeros(m,1);


tiempoFrankwolfeNonStep=zeros(m,1);
tiempoFrankwolfeStep=zeros(m,1);
tiempoInteriorPoint=zeros(m,1);
tiempoActSet=zeros(m,1);
tiempoQuadProg=zeros(m,1);




for j= inf:sup
    i=j-(inf-1);
    %Queremos usar el algoritmo interior-point
    options = optimoptions('fmincon', 'Algorithm', 'interior-point', 'MaxIterations', j, 'Display', 'off');
    tic;
    [~, fval] = fmincon(@(x)quadratic(x,H,b),X0,Bdesg,bdesg,[],[],[],[],[], options);

    tiempoIPNew(i) = toc;
    fvalIPNew(i)=fval;

    %Ahora vamos a utilizar Frank Wolfe sin paso óptimo
    tic;
    [~, fval2] = FWOptStep(H,b,Bdesg,bdesg, [],[],[],[],j,X0,0);

    tiempoFWNSNew(i)= toc;
    fvalFWNSNew(i)=fval2;

    %Ahora vamos a utilizar Frank Wolfe con paso óptimo
    tic;
    [~, fval3] = FWOptStep(H,b,Bdesg,bdesg,[],[],[],[],j,X0,1);
    tiempoFWSNew(i)= toc;
    fvalFWSNew(i)=fval3;

    %Ahora utilizaremos Active set
    options = optimoptions('fmincon', 'Algorithm', 'active-set', 'MaxIterations', j, 'Display', 'off');
    tic;
    [~, fval4] = fmincon(@(x)quadratic(x,H,b),X0,Bdesg,bdesg,[],[],[],[],[], options);

    tiempoASNew(i) = toc;
    fvalASNew(i)=fval4;

    %Ahora utilizaremos Quadprog
    %options = optimoptions('quadprog', 'Display', 'off');
    %tic;
    %[~,fval5] = quadprog(2*H,b,Bdesg,bdesg,[],[],[],[],X0,options);
    
    %tiempoQPNew(i)=toc;
    %fvalQNew(i)=fval5;



    %Vamos con el problema original
    options = optimoptions('fmincon', 'Algorithm', 'interior-point', 'MaxIterations', j, 'Display', 'off');
    tic;
    [~, fval] = fmincon(@(z)quadratic(z,H1,b1),X01,[],[],Aeq,beq,imin,imax,[], options);

    tiempoInteriorPoint(i) = toc;
    fvalInteriorPoint(i)=fval;

    %Ahora vamos a utilizar Frank Wolfe sin paso óptimo
    tic;
    [~,fval2]= FWOptStep(H1,b1,[],[],Aeq,beq,imin,imax,j,X01,0);
    %[~, fval2] = FWOptStep(H1,b1, ,Aeq,beq, imin,imax,j,X01,0);

    tiempoFrankwolfeNonStep(i)= toc;
    fvalFrankWolfeNonStep(i)=fval2;

    %Ahora vamos a utilizar Frank Wolfe con paso óptimo
    tic;
    [~, fval3] = FWOptStep(H1,b1,[],[],Aeq,beq, imin,imax,j,X01,1);
    tiempoFrankwolfeStep(i)= toc;
    fvalFrankWolfeStep(i)=fval3;

    %Ahora utilizaremos Active set
    options = optimoptions('fmincon', 'Algorithm', 'active-set', 'MaxIterations', j, 'Display', 'off');
    tic;
    [~, fval4] = fmincon(@(z)quadratic(z,H1,b1),X01,[],[],Aeq,beq,imin,imax,[], options);

    tiempoActSet(i) = toc;
    fvalActSet(i)=fval4;

    %Ahora utilizaremos Quadprog
    %options = optimoptions('quadprog', 'Display', 'off');
    %tic;
    %[~,fval5] = quadprog(2*H1,b1,[],[],Aeq,beq,imin,imax,X01,options);
    %tiempoQuadProg(i)=toc;
    %fvalQuadProg(i)=fval5;

end

%%
figure;

subplot(2, 1, 1); % 2 filas, 1 columna, primer subgráfico
%plot(inf:sup, fvalFWNSNew,'r');
%hold on;
plot(inf:sup, fvalFWSNew, 'magenta');
hold on;
plot(inf:sup, fvalIPNew, 'b');
hold on;
plot(inf:sup, fvalASNew, 'g');
hold on;
%plot(inf:sup,fvalFrankWolfeNonStep,'r--');
%hold on;
plot(inf:sup, fvalFrankWolfeStep, 'magenta--');
hold on;
plot(inf:sup, fvalInteriorPoint, 'b--');
hold on;
plot(inf:sup, fvalActSet, 'g--');

legend('Frank Wolfe Step Transformed', 'Interior Point Transformed', 'Active Set Transformed', 'Frank Wolfe Step', 'Interior Point', 'Active Set');

title('Comparación de métodos');
xlabel('Número de iteraciones');
ylabel('Valor de las funciones');

subplot(2, 1, 2); % 2 filas, 1 columna, segundo subgráfico
%plot(inf:sup, tiempoFWNSNew, 'r');
%hold on;
plot(inf:sup, tiempoFWSNew, 'magenta');
hold on;
plot(inf:sup, tiempoIPNew, 'b');
hold on;
plot(inf:sup, tiempoASNew, 'g');
hold on;
%plot(inf:sup, tiempoFrankwolfeNonStep, 'r--');
%hold on;
plot(inf:sup, tiempoFrankwolfeStep, 'magenta--');
hold on;
plot(inf:sup, tiempoInteriorPoint, 'b--');
hold on;
plot(inf:sup, tiempoActSet, 'g--');


legend('Frank Wolfe Step Transformed', 'Interior Point Transformed', 'Active Set Transformed', 'Frank Wolfe Step', 'Interior Point', 'Active Set');

title('Comparación de métodos');
xlabel('Número de iteraciones');
ylabel('Tiempo de ejecución');

%Figuras sueltas

figure;
%plot(inf:sup, fvalFWNSNew,'r');
%hold on;
plot(inf:sup, fvalFWSNew, 'magenta');
hold on;
plot(inf:sup, fvalIPNew, 'b');
hold on;
plot(inf:sup, fvalASNew, 'g');
hold on;
%plot(inf:sup,fvalFrankWolfeNonStep,'r--');
%hold on;
plot(inf:sup, fvalFrankWolfeStep, 'magenta--');
hold on;
plot(inf:sup, fvalInteriorPoint, 'b--');
hold on;
plot(inf:sup, fvalActSet, 'g--');

legend('Frank Wolfe Step Transformed', 'Interior Point Transformed', 'Active Set Transformed', 'Frank Wolfe Step', 'Interior Point', 'Active Set');

title('Comparación de métodos');
xlabel('Número de iteraciones');
ylabel('Valor de las funciones');

figure;

%plot(inf:sup, tiempoFWNSNew, 'r');
%hold on;
plot(inf:sup, tiempoFWSNew, 'magenta');
hold on;
plot(inf:sup, tiempoIPNew, 'b');
hold on;
plot(inf:sup, tiempoASNew, 'g');
hold on;
%plot(inf:sup, tiempoFrankwolfeNonStep, 'r--');
%hold on;
plot(inf:sup, tiempoFrankwolfeStep, 'magenta--');
hold on;
plot(inf:sup, tiempoInteriorPoint, 'b--');
hold on;
plot(inf:sup, tiempoActSet, 'g--');

legend('Frank Wolfe Step Transformed', 'Interior Point Transformed', 'Active Set Transformed', 'Frank Wolfe Step', 'Interior Point', 'Active Set');

title('Comparación de métodos');
xlabel('Número de iteraciones');
ylabel('Tiempo de ejecución');

%Gráficos conjuntos
figure;
subplot(2, 1, 1); % 2 filas, 1 columna, primer subgráfico
%plot(inf:sup, fvalFWNSNew,'r');
%hold on;
plot(inf:sup, fvalFWSNew, 'magenta');
hold on;
plot(inf:sup, fvalIPNew, 'b');
hold on;
plot(inf:sup, fvalASNew, 'g');

legend('Frank Wolfe Step Transformed', 'Interior Point Transformed', 'Active Set Transformed');
title('Comparación de métodos');
xlabel('Número de iteraciones');
ylabel('Valor de las funciones');

subplot(2, 1, 2); % 2 filas, 1 columna, primer subgráfico
%plot(inf:sup, tiempoFWNSNew, 'r');
%hold on;
plot(inf:sup, tiempoFWSNew, 'magenta');
hold on;
plot(inf:sup, tiempoIPNew, 'b');
hold on;
plot(inf:sup, tiempoASNew, 'g');


legend('Frank Wolfe Step Transformed', 'Interior Point Transformed', 'Active Set Transformed');
title('Comparación de métodos');
xlabel('Número de iteraciones');
ylabel('Tiempo de ejecución');


figure;
subplot(2, 1, 1); % 2 filas, 1 columna, primer subgráfico
%plot(inf:sup,fvalFrankWolfeNonStep,'r');
%hold on;
plot(inf:sup, fvalFrankWolfeStep, 'magenta');
hold on;
plot(inf:sup, fvalInteriorPoint, 'b');
hold on;
plot(inf:sup, fvalActSet, 'g');
legend('Frank Wolfe Step', 'Interior Point', 'Active Set');
title('Comparación de métodos');
xlabel('Número de iteraciones');
ylabel('Valor de las funciones');

subplot(2, 1, 2); % 2 filas, 1 columna, primer subgráfico
%plot(inf:sup, tiempoFrankwolfeNonStep, 'r');
%hold on;
plot(inf:sup, tiempoFrankwolfeStep, 'magenta');
hold on;
plot(inf:sup, tiempoInteriorPoint, 'b');
hold on;
plot(inf:sup, tiempoActSet, 'g');

legend('Frank Wolfe Step', 'Interior Point', 'Active Set');
title('Comparación de métodos');
xlabel('Número de iteraciones');
ylabel('Tiempo de ejecución');


