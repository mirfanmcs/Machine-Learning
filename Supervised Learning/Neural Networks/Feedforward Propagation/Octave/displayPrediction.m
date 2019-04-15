function displayPrediction(Theta1, Theta2, X)

m = size(X, 1);

%Randomly permute examples
rp = randperm(m);



for i = 1:m
      % Display 
      fprintf('\nDisplaying Example Image\n');
      
	  displayData(X(rp(i), :));

      pred = predict(Theta1, Theta2, X(rp(i),:));
      fprintf('\nNeural Network Prediction: %d (digit %d)\n', pred, mod(pred, 10));
      
      % Pause with quit option
      s = input('Paused - press enter to continue, q to exit:','s');
      if s == 'q'
        break
      end
end

end
