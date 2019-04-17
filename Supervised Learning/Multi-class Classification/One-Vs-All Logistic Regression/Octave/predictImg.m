function p = predictImg(Theta, Img)
    %imfinfo(Img)
    X = imread(Img); % reads the image (20x20)

    X = double(X);% converts it to double
    temp = X;% creates a copy for later use

    X = (X.-128)./255;%normalize the features
    X = X .* (temp > 0);%return the original 0 values to the X
    X = reshape(X, [], numel(X));%converts the 20x20 matrix into a 1x400 vector

    displayData(X);%display the image imported

    p = predictOneVsAll(Theta, X)

    if p==10
        p=0;
    end

    fprintf('\nImage is number: %d\n', p);

end