function plotSigmoid()

z = [-10 -9 -8 -7 -6 -5 -4 -3 -2 -1 0 1 2 3 4 5 6  7 8 9 10];

figure;
plot(z, sigmoid(z))
hold on;
plot(linspace(0,0,21),linspace(0,1,21),'k');  % add vertical line
xlabel('z');
ylabel('g(z)');
title('Sigmoid');
hold off;

end
