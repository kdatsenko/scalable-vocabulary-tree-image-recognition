function [H_transf_best, highest_num_inliers, best_SSD] = ...
    ransac_homography(ref_im, test_im, num_trials, k)

if nargin<4
    k = Inf; %all keypoint matches
end;

ref_img = single(ref_im);
test_img = single(test_im);
%compute the SIFT frames (keypoints) and descriptors
[f_ref,d_ref] = vl_sift(ref_img) ;
[f_test,d_test] = vl_sift(test_img) ;

[matches, scores] = vl_ubcmatch(d_ref, d_test, 1.5);

num_matches = size(matches, 2);
matches_joined = zeros(num_matches, 3);
matches_joined(:, 1) = scores';
matches_joined(:, 2:3) = matches';
scored_matches = sortrows(matches_joined);
matches_ref = scored_matches(:, 2);
matches_test = scored_matches(:, 3);

%top k matches
if num_matches > k
    num_matches = k; %limit on top matches
end

threshold = 3;
highest_num_inliers = 0; %size of largest set of inliers
H_transf_best = 0;
best_SSD = Inf;

if (num_matches < 4) %cannot do a homography, exit
	return;
end

for i=1:num_trials %set as input to the RANSAC algo.
    sample = randsample(num_matches, 4);
    
    
    %compute affine transf, get transf matrix
    sr = matches_ref(sample);
    st = matches_test(sample);
    
    H_transf = homo_transf(f_ref, f_test, sr, st);
    num_inliers = 0;
    SSD = 0;   
    for j=1:num_matches
        %The matrix f has a column for each keypoint. 
        %A keypoint has center f(1:2)
        orig_point = f_ref(1:2, matches_ref(j));
        x = orig_point(1);
        y = orig_point(2);
        P_ref = [x; y; 1];
        Pt_h = H_transf * P_ref; %homogenous coords
        Pt = [Pt_h(1)/Pt_h(3), Pt_h(2)/Pt_h(3)]; %normal coords
        squared_diff = (Pt(1) - f_test(1, matches_test(j)))^2 + ...
                       (Pt(2) - f_test(2, matches_test(j)))^2;
        
        dist = sqrt(squared_diff);
        if (dist <= threshold)
            num_inliers = num_inliers + 1;
        end
        SSD = SSD + sqrt(squared_diff); 
    end
    mean_SSD = SSD / num_matches;
    if num_inliers > highest_num_inliers
        H_transf_best = H_transf;
        highest_num_inliers = num_inliers;
        best_SSD = mean_SSD;
    end 
end

end