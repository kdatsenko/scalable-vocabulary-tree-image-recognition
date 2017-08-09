function visualize_homography(test_img, r, c, H)

%[r, c, ~] = size(test_img); 

%Expects a 3x3 H matrix

% create a matrix with the homogenous coordinates of the four 
% corners of the current image
pt_matrix = zeros(3, 4); %each column is a corner point
pt_matrix(:, 1) = [1,1,1]; %top-left 
pt_matrix(:, 2) = [c,1,1]; %bottom-left
pt_matrix(:, 3) = [1,r,1]; %top-right
pt_matrix(:, 4) = [c,r,1]; %bottom-right
result = H*pt_matrix;
result(1,:) = result(1,:) ./ result(3,:); %space coordinates
result(2,:) = result(2,:) ./ result(3,:);

% Plot lines for dvd
figure(4);
imshow(test_img);
hold on;

line([result(1, 1),result(1, 2)],[result(2, 1),result(2, 2)],'Color','g', 'Linewidth', 4);
line([result(1, 1),result(1, 3)],[result(2, 1),result(2, 3)],'Color','g', 'Linewidth', 4);
line([result(1, 3),result(1, 4)],[result(2, 3),result(2, 4)],'Color','g', 'Linewidth', 4);
line([result(1, 2),result(1, 4)],[result(2, 2),result(2, 4)],'Color','g', 'Linewidth', 4);

hold off;

end
