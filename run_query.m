function [candidates, scores]=run_query(query_img, vocabulary_tree, ...
                        invfindex, img_norms, node_weights, improvement)

K = vocabulary_tree.K;
L = vocabulary_tree.depth;
nnodes = length(node_weights);
Nimgs = length(img_norms);

%compute the keypoints and descriptors for the query image
if improvement == 0 %no improvement, DoG, SIFT
    [~, sift_desc] = vl_sift(query_img);
elseif improvement == 1 %Affine Invariant Features, SIFT
    [~, sift_desc] = vl_covdet(query_img, 'Method', 'HarrisLaplace',  ...
                              'EstimateAffineShape', true);
    sift_desc = sift_desc * 255;
    sift_desc = uint8(sift_desc);
else %improvement = 2, descriptors with spatial context features
    [frames, sift_desc] = vl_sift(query_img);
    sift_desc=descr_spatial_stats(sift_desc, frames);
end

fprintf('Finished SIFT computation\n');

number_of_descriptors=size(sift_desc,2);

q_vector = zeros(1, nnodes); %single row vector
%
% GET entire query vector, and afterwards normalize it
%
for i=1:number_of_descriptors
	
    sift_desc_i = sift_desc(:, i); %get one sift descriptor
    path_to_leaf =descr_to_path(sift_desc_i, vocabulary_tree, L);

	%get index based on path parts
    index=0; %node index
    for j=1:numel(path_to_leaf) %numel is number of array elements
        index= index*K+path_to_leaf(j); %node along the path
        q_vector(index) = q_vector(index) + 1; %increment ni
    end
    %if mod(i, 100) == 0
    %    fprintf('Finished computing for desc %d out of %d\n', i, number_of_descriptors);
    %end
end

weighted_q_vector = q_vector .* node_weights;
q_vector_norm = norm(weighted_q_vector, 1); %L1 norm

fprintf('Finished computing the normalized q vector\n');

%--------------------------------
% Simple Test Case
%
% Nimgs = 3;
% nnodes = 14;
% K=2;
% L=3;
% nleaves = K^L;
% invfindex(nleaves).images=[];
% invfindex(nleaves).mi=[];
% 
% invfindex(1).images=[1 2 3];
% invfindex(1).mi=[1 2 3];
% 
% invfindex(2).images=[1];
% invfindex(2).mi=[4];
% 
% invfindex(3).images=[];
% invfindex(3).mi=[];
% 
% invfindex(4).images=[3 1];
% invfindex(4).mi=[7 8];
% 
% invfindex(5).images=[3];
% invfindex(5).mi=[9];
% 
% invfindex(6).images=[1 3 2];
% invfindex(6).mi=[10 11 12];
% 
% invfindex(7).images=[2 1];
% invfindex(7).mi=[1 3];
% 
% invfindex(8).images=[2 3];
% invfindex(8).mi=[4 5];
% 
% node_Ni = [3 3 3 2 3 3 3 1 0 2 1 3 2 2];
% 
% 
% q_vector = [6 10 1 5 5 5 1 0 2 3 4 1 0 5];
% 
% node_weights = log((Nimgs)./node_Ni);
% node_weights(node_Ni == 0) = 0;
% node_weights
% 
% weighted_q_vector = q_vector .* node_weights;
% 
% q_vector_norm = norm(weighted_q_vector, 1);
% 
% normalized_q = weighted_q_vector / q_vector_norm;
% 
% im1 = [13 13 5 8 10 3 1 4 0 8 0 10 3 0];
% im2 = [2 17 2 0 12 5 2 0 0 0 0 12 1 4];
% im3 = [10 25 3 7 20 5 3 0 0 7 9 11 0 5];
% 
% 
% 
% wim1 = im1 .* node_weights;
% n_wim1 = wim1 / norm(wim1, 1);
% wim2 = im2 .* node_weights;
% n_wim2 = wim2 / norm(wim2, 1);
% wim3 = im3 .* node_weights;
% n_wim3 = wim3 / norm(wim3, 1);
% 
% img_norms = [norm(wim1, 1), norm(wim2, 1), norm(wim3, 1)];
% 
% scores_official = zeros(1, 3);
% scores_official(1) = norm(normalized_q - n_wim1, 1);
% scores_official(2) = norm(normalized_q - n_wim2, 1);
% scores_official(3) = norm(normalized_q - n_wim3, 1);
% 
% 
% scores_unofficial = zeros(1, 3);
% scores_unofficial = scores_unofficial + 2;
% for i=1:nnodes
%     if (wim1(i) ~= 0 && weighted_q_vector(i) ~= 0)
%         a = abs(normalized_q(i) - n_wim1(i)) - abs(normalized_q(i)) - abs(n_wim1(i));
%         scores_unofficial(1) = scores_unofficial(1) + a;
%     end
% end
% for i=1:nnodes
%     if (wim2(i) ~= 0 && weighted_q_vector(i) ~= 0)
%         a = abs(normalized_q(i) - n_wim2(i)) - abs(normalized_q(i)) - abs(n_wim2(i));
%         scores_unofficial(2) = scores_unofficial(2) + a;
%     end
% end
% for i=1:nnodes
%     if (wim3(i) ~= 0 && weighted_q_vector(i) ~= 0)
%         a = abs(normalized_q(i) - n_wim3(i)) - abs(normalized_q(i)) - abs(n_wim3(i));
%         scores_unofficial(3) = scores_unofficial(3) + a;
%     end
% end
% 
% scores_official
% scores_unofficial
%--------------------------------


normalized_diff_scores = zeros(1, Nimgs); %score for each img
normalized_diff_scores = normalized_diff_scores + 2;

%last node number of second last level of nodes (+1 for first leaf node)
lastlvl_cond = nnodes - (K^L);

img_indx_table = zeros(1, Nimgs); %fast index into small
curr_node_inverted_file = zeros(2, Nimgs); %max size
process_parent = false; %true if all children of this node were already visited

% Storage container for accumlated mi for d-vectors
% Computes virtual inverted files bottom up from first from unions 
% of the inverted files, then from generated inverted files from inner nodes
% Recycles inner node files after they are used
% At any moment, space used by mi_accumulator is no pgreater than space
% for original virtual inverted file structure (INVFINDEX)
mi_accumulator = cell(1, nnodes); 

%
% DO Bottom-Up Scoring Procedure 
% Reconstructs the Virtual inverted files from the leaves
% as described in Section 5 of Nister et al. 
%

current_node = 1;
while current_node > 0 %not before the 1, 2, blah blah .... (all the way to the dummy root)
	%fprintf('processing current node %d %d\n', current_node, q_vector(current_node));
	have_processed = false;
	if q_vector(current_node) ~= 0 %qi != 0
		if process_parent == false % all children have not yet been processed 

%% LEAF CASE ----------------------------------------------------------------------------
			first_child = (current_node * K) + 1;
			if ~(first_child > lastlvl_cond) %children are not leaf nodes
				%go to children (post-order traversal)
				current_node = first_child;
				continue;
			else % first_child > lastlvl_cond; this is the parent of a leaf node
				%deal with leaf nodes (INVFINDEX)
    
				%making a new inverted file
                %for rewriting, equivalent of resetting all img_indx_table to zeros
                %for rewriting, how many imgs were mapped for this current node
				img_indx_size = 0; %size of curr_node_inverted_file

				last_child = first_child+K-1;
				for c=first_child:last_child %leaf node index

					l = c - lastlvl_cond; %special leaf index only into invfindex
					%union operation for each inverted list for the leaves
					for im=1:length(invfindex(l).images) 
						id = invfindex(l).images(im);
						inv_i = img_indx_table(id); %stored inverted list index (union)
						%inv_i = 0 - allocate for entirely new img
						%inv_i > img_indx_size - detected outdated info
						if inv_i > img_indx_size || inv_i == 0 || id ~= curr_node_inverted_file(1, inv_i) 
							inv_i = img_indx_size + 1;
							img_indx_size = img_indx_size + 1;
							img_indx_table(id) = inv_i; %write over
							curr_node_inverted_file(1, inv_i) = id; %write over
							curr_node_inverted_file(2, inv_i) = invfindex(l).mi(im);
						else %accumulate mi scores for same img=id over all children
							cur_val = curr_node_inverted_file(2, inv_i);
							curr_node_inverted_file(2, inv_i) = cur_val + invfindex(l).mi(im);
                        end	
                        
                        %ACCUMULATE |qi - di| - |qi| - |di| for each LEAF node!
                        if weighted_q_vector(c) ~= 0
                            qi = weighted_q_vector(c) / q_vector_norm;
                            di_unnormalized = invfindex(l).mi(im) * node_weights(c);
                            if di_unnormalized == 0
                                continue;
                            end
                            di = di_unnormalized / img_norms(id);
                            diff_i = abs(qi - di) - abs(qi) - abs(di); %L1 norm
                            %initial val 2 was pre-added
                            normalized_diff_scores(id) = normalized_diff_scores(id) + diff_i; 
                        end                        
					end	%end imgs for this leaf node c	                        
				end %end all leaves for this parent (parent is current node)

				%add curr_node_inverted_file to cell array
				mi_accumulator{current_node} = curr_node_inverted_file(:, 1:img_indx_size);

				%end parent with leafs CASE
				have_processed = true; 
			end

%% REGULAR CASE ----------------------------------------------------------------------------
		else %if process_parent == true

			process_parent = false; %reset
			img_indx_size = 0; %making a new small db list, reset all img_indx_table to zeros

			first_child = (current_node * K) + 1;
			last_child = first_child+K-1;
			for c=first_child:last_child %for each child of parent
          
                %union operation for each inverted list for the children
                for im=1:size(mi_accumulator{c}, 2) %cols
                    id = mi_accumulator{c}(1, im); %ID of image
                    inv_i = img_indx_table(id); %stored inverted list index (union)
                    %inv_i = 0 - allocate for entirely new img
                    %inv_i > img_indx_size - detected outdated info
                    if inv_i > img_indx_size || inv_i == 0 || id ~= curr_node_inverted_file(1, inv_i)
                        inv_i = img_indx_size + 1;
                        img_indx_size = img_indx_size + 1;
                        img_indx_table(id) = inv_i; %write over
                        curr_node_inverted_file(1, inv_i) = id; %write over
                        curr_node_inverted_file(2, inv_i) = mi_accumulator{c}(2, im);
                    else %accumulate mi scores for same img=id over all children
                        cur_val = curr_node_inverted_file(2, inv_i);
                        curr_node_inverted_file(2, inv_i) = cur_val + mi_accumulator{c}(2, im);
                    end	
                end	%end imgs for this child node
                mi_accumulator{c} = []; %delete inverted file of child to save space	
			end %end all child nodes for this parent

			%add curr_node_inverted_file to cell array
			mi_accumulator{current_node} = curr_node_inverted_file(:, 1:img_indx_size);
			have_processed = true;
		end % if process_parent == true

%----------------------------------------------------------------------------
	end %END OF PROCESSING THE NODE

	if have_processed == true %concatenation of inverted lists of children done        
        
		%accumulate normalized difference scores for this node i
        if weighted_q_vector(current_node) ~= 0
            qi = weighted_q_vector(current_node) / q_vector_norm;
            Ni = size(mi_accumulator{current_node}, 2);
            %every img with at least one descriptor vector path through current_node
            for i=1:Ni
                id = mi_accumulator{current_node}(1, i);
                mi = mi_accumulator{current_node}(2, i);
                
                di_unnormalized = mi * node_weights(current_node);
                if di_unnormalized == 0
                    continue;
                end
                di = di_unnormalized / img_norms(id);

                diff_i = abs(qi - di) - abs(qi) - abs(di);
                normalized_diff_scores(id) = normalized_diff_scores(id) + diff_i; %2 was already added
            end
        end
	end

	%Process sibling
	w = current_node / K;
	if (round(w) ~= w) %is there a sibling on this level?
		current_node = current_node + 1; %next sibling
	else %this is the last node of branches
		%finished w/ children, now go back up and process parent
		process_parent = true; 
		current_node = floor((current_node - 1) / K); %parent node
	end

end %END of Bottom-up scoring procedure

%% Extract top 10 candidates

candidates = zeros(1, 10);
scores = zeros(1,10);

j = 1;
%extract top ten (only need a partial sort, however)
while j <= 10
	[score, img_id] = min(normalized_diff_scores);
	candidates(j) = img_id;
	scores(j) = score;
	normalized_diff_scores(img_id) = inf;
	j = j + 1;
end


end