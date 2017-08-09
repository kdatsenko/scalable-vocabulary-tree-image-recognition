% Solve for the Homography transformation between using 4 matched features
function H_mat = homo_transf(f_im1, f_im2, ind1, ind2)

% We assume we are using 4 keypoint matches 
k = 4;
A = zeros(2*k, 9);

for i = 1:k  
    x1 = f_im1(1, ind1(i));
    y1 = f_im1(2, ind1(i));
    x2 = f_im2(1, ind2(i));
    y2 = f_im2(2, ind2(i));
    
    A(2*(i-1) + 1, :) = [x1 y1 1 0 0 0 -1*x2*x1 -1*x2*y1 -1*x2];
    A(2*(i-1) + 2, :) = [0 0 0 x1 y1 1 -1*y2*x1 -1*y2*y1 -1*y2];
end

%returns diagonal matrix D of eigenvalues and matrix V whose 
%columns are the corresponding right eigenvectors
%[V,D] = eig(A)

M = A'*A;
[V, D] = eig(M); %eigenvalues
min_eval = Inf;
min_i = 0;
for i=1:size(V, 2) %eigenvectors
    if D(i,i) < min_eval %find min eigenvalue
        min_eval = D(i,i);
        min_i = i;
    end
end

H_vect = V(:, min_i); %eigenvector w/ min eigenvalue
H_mat = zeros(3, 3);
H_mat(1, 1:3) = H_vect(1:3);
H_mat(2, 1:3) = H_vect(4:6);
H_mat(3, 1:3) = H_vect(7:9);

end