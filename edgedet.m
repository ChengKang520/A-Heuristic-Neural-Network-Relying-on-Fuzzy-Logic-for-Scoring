
function J = edgedet(I, N)

% undecimated wavelet transform
[approx, detail] = a_trous_dwt(I, N);

% module of the wavelet detail coefficients
D = abs(detail(:,:,N));

% segment the modules coeffiient matrix
J = (D > filter2(ones(3)/9, D)) .* (D > mean2(D));

% due to the 5x5 convolution kernel and the 3x3 smoothing operation, zero
% out the margins
[R C] = size(J);
J(1:3, :) = 0;
J(R-2:R, :) = 0;
J(:, 1:3) = 0;
J(:, C-2:C) = 0;

    function I = a_trous_idwt(I, N)
        
        I = A + sum(D, 3); % sum along the 3rd dimension
        
    end


    function [A, D] = a_trous_dwt(I, N)
        % D2 = D(:,:,2); % extract the second plane of the 3D array
        
        
        B3 = [1/16, 1/4, 3/8, 1/4 1/16];
        h = B3' * B3;
        
        A = double(I);
        for level = 1:N
            approx(:,:,level) = conv2(A, h, 'same');
            D(:, :, level) = A - approx(:, :, level);
            A = approx(:, :, level);
        end
        
    end


end


