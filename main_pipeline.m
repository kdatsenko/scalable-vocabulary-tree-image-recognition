function main_pipeline()

%Setup: data folders
db_dir = 'dvd_covers/Reference';
test_dir='dvd_covers/Canon';

queries = [3]; %Set list of query images

%Init: tree parameters
k=10; %branching factor
L=6; %number of levels of the tree (excluding root node)
show_candidates = true;

%Part 6: Come up with and implement your own improvements
%0 - no improvement, DoG, SIFT
%1 - Harris-Affine Invariant Features, SIFT
%2 - SIFT Descriptors with spatial context features concatenated on the end
improvement =0;


%
% Part 3(c): Hierarchical Vocabulary tree generation
%
if ~exist('vocabulary_tree.mat')
	vocabulary_tree=build_vocabulary_tree(db_dir, k, L, improvement);
	fprintf('Finished Vocab tree creation.\n');
	%save vocab tree as a file....
	save('vocabulary_tree.mat','vocabulary_tree');
else 
	load('vocabulary_tree.mat', 'vocabulary_tree');    
end



% Computing the Virtual Inverted File Index
% 
% Precomputing node weights (entropy)
% Precomputing norms of d-vectors representing DB images
if ~exist('invfindex.mat')
	[invfindex, img_norms, node_weights, dbImgNames]=inverted_file_index(db_dir, vocabulary_tree, improvement);
	%save db_vectors as a file....
	save('invfindex.mat','invfindex');
	save('img_norms.mat','img_norms');
	save('node_weights.mat','node_weights');
	%save the order of imgs and their ids (indices in dbImgNames) just in case
	%order read changes, or an image gets lost
	save('dbImgNames.mat','dbImgNames'); 
else
	load('invfindex.mat','invfindex');
	load('img_norms.mat','img_norms');
	load('node_weights.mat','node_weights');
	load('dbImgNames.mat','dbImgNames');
end



%
% Part 4(a): Retrieval of top ten matches with vocabulary tree
%
for l=1:numel(queries)
    test_image_num=queries(l); %First image in directory
    testImgNames = dir([db_dir '/*.jpg']);
    test_img_color=imresize(imread(fullfile(test_dir,testImgNames(test_image_num).name)), 0.25);
    fprintf('\nTest image %d: %s\n',test_image_num,testImgNames(test_image_num).name);
    test_img = single(rgb2gray(test_img_color));
    
    [candidates,scores]=run_query(test_img, vocabulary_tree, invfindex, img_norms, node_weights, improvement);
    
    if any(candidates == test_image_num)
        fprintf('Position found %d\n',find(candidates==test_image_num));
    else
        fprintf('FAILED THIS CASE %d\n',test_image_num);
    end
end

if (show_candidates)
    %Show first 9 candidates
    figure(1), clf;
    imagesc(test_img_color);
    axis off; 
    set(gcf, 'color', 'white');
    title(['Image ',num2str(test_image_num)]);
    display_candidates(db_dir, candidates, scores);
end

%
% Part 4(b): Compute Homography with RANSAC for each of the 10 candidates
% Part 5(a): Find the DVD cover with the highest number of inliers. Plot it.
%
num_trials = 500;
highest_num_inliers = -inf;
for i=1:10
	ref_img_color = imread(fullfile(db_dir,dbImgNames(candidates(i)).name));
	ref_img=rgb2gray(ref_img_color);
	[H, inliers] = ransac_homography(ref_img, test_img, num_trials, 100);
	if inliers > highest_num_inliers
		best_cover_ind = i;
		best_cover_img = ref_img_color;
		H_transf_best = H;
		highest_num_inliers = inliers;
	end
end

figure(3), clf;
imshow(best_cover_img);

%
% Plot the test image with the localized DVD cover
%
[r, c, ~] = size(best_cover_img);
visualize_homography(test_img_color, r, c, H_transf_best);



end