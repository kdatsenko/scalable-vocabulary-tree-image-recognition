function path_to_leaf=descr_to_path(descriptor, vocabulary_tree, depth_of_tree)

path_to_leaf = zeros(depth_of_tree, 1);
node = vocabulary_tree;
for level=1:depth_of_tree
    best_center = 1; 
    best_center_score = inf;
    for branch=1:size(node.centers, 2)
        %grab cluster center for this branch
        compare_center = node.centers(:, branch);
        
        %get branch with min euclidean dist
        dotprod = single(compare_center) - single(descriptor);
        dotprod = sum(dotprod.^2);
        
        %get branch with max dot prod
        %dotprod = dot(single(compare_center), single(descriptor));
        %dotprod = dotprod / norm(single(compare_center), 2);
        %dotprod = dotprod / norm(single(descriptor), 2);
        
        if dotprod < best_center_score
            best_center = branch;
            best_center_score = dotprod;
        end
    end
    path_to_leaf(level) = best_center;
    if level ~= depth_of_tree %leaf level
        node = node.sub(best_center);
    end
end