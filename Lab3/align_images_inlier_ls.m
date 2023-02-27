function warped = align_images_inlier_ls(source, target, threshold, upright)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
src_points = detectSIFTFeatures(source);
% src_points = detectSURFFeatures(source);
[src_features, src_validPoints] = extractFeatures(source,src_points,'Upright', upright);

trg_points = detectSIFTFeatures(target);
% trg_points = detectSURFFeatures(target);
[trg_features, trg_validPoints] = extractFeatures(target,trg_points,'Upright', upright);

corrs = matchFeatures(src_features,trg_features);

src_points = src_validPoints.Location(corrs(:,1),:)';

trg_points = trg_validPoints.Location(corrs(:,2),:)';

[A,t] = ransac_fit_affine_(src_points, trg_points, threshold);
[A1,t1] = ransac_fit_affine(src_points, trg_points, threshold);
res1 = residual_lgths(A, t, src_points, trg_points);
res2 = residual_lgths(A1, t1, src_points, trg_points);
sres1 = sum(res1);
sres2 = sum(res2);
sres1,sres2
warped = affine_warp(size(target), source, A, t);
end
function [A,t] = ransac_fit_affine_(pts, pts_tilde, threshold)
k_max = size(pts, 2) * 1000;
outliers = size(pts, 2);

nu = 0.01;

count = 0;
while count < k_max
    i = randperm(size(pts, 2), 3);
    [A_c, t_c] = least_squares_affine(pts(:, i), pts_tilde(:, i));
    
    if (~isnan(A_c) & ~isinf(A_c) & ~isnan(t_c) & ~ isinf(t_c))
        residual = residual_lgths(A_c, t_c, pts, pts_tilde);
        outliers_c = sum(residual > threshold);

        if (outliers > outliers_c)
            outliers = outliers_c;
            A = A_c;
            t = t_c;
            
            %Update k_max
            ep = (size(pts, 2) - outliers) / size(pts, 2);
            k_max = log(nu) / log(1-ep^3);
        end
    end
    count = count + 1;
end

end
