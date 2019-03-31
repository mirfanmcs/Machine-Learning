function plotSigmoid()

z = linspace(-10,10,21);

figure;
plot(z, sigmoid(z))
hold on;
plot(linspace(0,0,21),linspace(0,1,21),'k');  % add vertical line
xlabel('z');
ylabel('g(z)');
title('Sigmoid');
hold off;

end
