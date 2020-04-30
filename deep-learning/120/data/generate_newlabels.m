load('test_data.mat');
load('train_data.mat');

newtrain = [];
newlabels = dataset;

num_samples = 1;
for a=1:length(labels)
    disp(a);
    defect = labels(a,:,:,:) - dataset(a,:,:,:);
    defect_3dlocs = get3dLocs(defect);
    try
        [k,av] = convhull(defect_3dlocs);
        convhull3d = [defect_3dlocs(k,1), defect_3dlocs(k,2), defect_3dlocs(k,3)];

        contour = [];
        count = 1;
        dataset_3dlocs = get3dLocs(dataset(a,:,:,:));
        for b=1:length(convhull3d)
            nearestIdx = knnsearch(dataset_3dlocs,convhull3d(b,:));
            if isempty(contour) || ~nnz(ismember(contour, dataset_3dlocs(nearestIdx,:), 'rows'))
                contour(count,:) = dataset_3dlocs(nearestIdx,:);
                count = count + 1;
            end
        end

        for b=1:length(contour)
            newlabels(num_samples,contour(b,1),contour(b,2),contour(b,3)) = 2; % label as contour
        end
        newtrain(num_samples,:,:,:) = dataset(a,:,:,:);
        num_samples = num_samples + 1;
    catch ME
        continue;
    end
end

disp(num_samples);
save('train120.mat', 'newtrain');
save('labels120.mat', 'newlabels');

