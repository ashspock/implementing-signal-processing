clc;
clear;
clear all;
%%
% Define the path to the folder containing subfolders with face images
main_folder = 'C:\Users\sachin\Documents\MATLAB\trainData';

% Get a list of all subfolders (each containing face images of one individual)
subfolders = dir(main_folder);
subfolders = subfolders([subfolders.isdir]); % Keep only directories
subfolders = subfolders(~ismember({subfolders.name}, {'.', '..'})); % Remove '.' and '..' directories

% Initialize cell arrays to store images and corresponding labels
trainImages = {};
trainLabels = {};

% Iterate through each subfolder
for i = 1:length(subfolders)
    subfolder_name = subfolders(i).name;
    subfolder_path = fullfile(main_folder, subfolder_name);

    % Get a list of all image files in the subfolder
    image_files = dir(fullfile(subfolder_path, '*.pgm')); % Assuming images are in JPEG format, modify if needed

    % Iterate through each image file
    for j = 1:length(image_files)
        image_file = image_files(j).name;
        image_path = fullfile(subfolder_path, image_file);

        % Read the image
        img_test = imread(image_path);

        % Convert the image to grayscale (if necessary)
        if size(img_test, 3) > 1
            img_test = rgb2gray(img_test);
        end
        %img = imresize(img,0.5);
        % Store the image and label
        trainImages{end+1} = img_test;
        trainLabels{end+1} = subfolder_name;
    end
end
% Display the number of imported images and labels
fprintf('Number of images imported: %d\n', length(trainImages));
fprintf('Number of labels: %d\n', length(trainLabels));

% Now, you can use the 'images' and 'labels' cell arrays for face recognition using eigenfaces
%%  2
% Convert images to a matrix where each column represents a flattened image

num_train_images = length(trainImages);
image_size = size(trainImages{1});
image_vector_length = prod(image_size); % Number of pixels in each image
image_matrix = zeros(image_vector_length, num_train_images);
for i = 1:num_train_images
    image_vector = reshape(trainImages{i}, [], 1); % Flatten the image
    image_matrix(:, i) = double(image_vector'); % Store the flattened image as a column in the image matrix
end
%%  3
% Compute mean face
mean_face = mean(image_matrix,2);
%%
% display mean face
figure;
imshow(reshape(mean_face, image_size), []);
title('Mean Face')
%%  4
% Subtract mean face from each image
%centered_image_matrix = image_matrix - repmat(mean_face, 1, num_images);
centered_image_matrix = image_matrix - mean_face;
%%  5
% Perform PCA
covariance_matrix = cov(centered_image_matrix');%centered_image_matrix * centered_image_matrix';
[eigen_vectors, eigen_values] = eig(covariance_matrix);
%[U, S, ~] = svd(centered_image_matrix', 'econ'); % Perform singular value decomposition (SVD)
%num_components = 10; % Choose the number of principal components (eigenfaces) to keep
%eigenfaces = U(:, 1:num_components); % Select top eigenfaces
%%  6
% Sort eigen vectors in descending order of eigenvalues
[eigenValuesSorted , sorted_indices] = sort(diag(eigen_values), 'descend');
figure;
% eigenValuesSorted = diag(eigen_values)
scatter(1:1:100, eigenValuesSorted(100,1));
sorted_eigen_vectors = eigen_vectors(:, sorted_indices);
title('Top 100 eigen values');
%%  7
% Keep only the top k eigenfaces (eigen vectors with highest eigenvalues)
num_components = 10;
top_eigen_vectors = sorted_eigen_vectors(:, 1:num_components);
%top_eigen_vectors = 255 - abs(top_eigen_vectors(:,:));
% Display eigenfaces

figure;
for i = 1:num_components
    subplot(2, 5, i);
    eigenface = reshape(top_eigen_vectors(:, i), image_size);
    imshow(eigenface,[]); % Assuming grayscale images, convert to uint8 for display
    title(['Eigenface ' num2str(i)]);
end
%% 8 Select top-k eigenfaces
k=100;
selectedEigenFaces = sorted_eigen_vectors(:,1:k);
%% 9 Project Training Images onto Eigenfaces
projectedTrainImages = selectedEigenFaces' * centered_image_matrix;

%% 1 Testing
% Define the path to the folder containing subfolders with testing face images
testing_main_folder = 'C:\Users\sachin\Documents\MATLAB\testData';

% Get a list of all subfolders (each containing face images of one individual)
testing_subfolders = dir(testing_main_folder);
testing_subfolders = testing_subfolders([testing_subfolders.isdir]); % Keep only directories
testing_subfolders = testing_subfolders(~ismember({testing_subfolders.name}, {'.', '..'})); % Remove '.' and '..' directories

% Initialize cell arrays to store testing images and corresponding labels
testing_images = {};
testing_labels = {};

% Iterate through each subfolder
for i = 1:length(testing_subfolders)
    subfolder_name = testing_subfolders(i).name;
    subfolder_path = fullfile(testing_main_folder, subfolder_name);

    % Get a list of all image files in the subfolder
    testing_image_files = dir(fullfile(subfolder_path, '*.pgm')); % Assuming images are in JPEG format, modify if needed

    % Iterate through each image file
    for j = 1:length(testing_image_files)
        image_file = testing_image_files(j).name;
        image_path = fullfile(subfolder_path, image_file);

        % Read the image
        img_test = imread(image_path);

        % Convert the image to grayscale (if necessary)
        if size(img_test, 3) > 1
            img_test = rgb2gray(img_test);
        end

        % Store the image and label
        testing_images{end+1} = img_test;
        testing_labels{end+1} = subfolder_name;
    end
end

% Display the number of imported testing images and labels
fprintf('Number of testing images imported: %d\n', length(testing_images));
fprintf('Number of testing labels: %d\n', length(testing_labels));

% Now, you can use the 'testing_images' and 'testing_labels' cell arrays for testing the face recognition algorithm
%% 2
% Project training and testing dataset onto eigenfaces
% projected_training = eigenfaces' * image_matrix';
projected_training = selectedEigenFaces' * image_matrix;

%%  3
num_test_images = length(testing_images);
num_test_subjects = numel(testing_subfolders);
num_test_images_per_subject = num_test_images/num_test_subjects;
test_image_size = size(testing_images{1});
test_image_vector_length = prod(test_image_size); % Number of pixels in each image
test_image_matrix = zeros(test_image_vector_length, num_test_images);
for i = 1:num_test_images
    test_image_vector = reshape(testing_images{i}, [], 1); % Flatten the image
    test_image_matrix(:, i) = (test_image_vector); % Store the flattened image as a column in the image matrix
end

%%  4

test_centered_matrix = (test_image_matrix - mean_face);
projected_testing = selectedEigenFaces'* test_centered_matrix;

%%  5
confusion_matrix = zeros(num_test_subjects,num_test_subjects);%num_classes,num_classes);

for i=1:num_test_subjects%num_classes
    for j=1:num_test_images_per_subject%num_test_images
        % distance = sqrt(sum((projectedImages - projected_testing(:, (i-1) * num_test_images + j)).^2, 1));
        distance = sqrt(sum((projectedTrainImages - projected_testing(:,(i-1)*num_test_images_per_subject+j)).^2, 1));
        % index of closest match
        [~,index] = min(distance);
        % identify recognized individual
        identifiedPerson = ceil(index/7);
        % confusion matrix
        confusion_matrix(i,identifiedPerson) = confusion_matrix(i,identifiedPerson)+1;
    
        % Display the results
        figure;
        subplot(1, 2, 1); 
        imshow(uint8(reshape(test_centered_matrix(:,(i - 1) * num_test_images_per_subject+ j) + mean_face, image_size))); 
        title(['Test Image of person',num2str(i)]);
        subplot(1, 2, 2); 
        imshow(uint8(reshape(image_matrix(:, index), image_size))); 
        title(['Recognized as Person ', num2str(identifiedPerson)]);    
    end
end

%% Plot confusion matrix
figure;
confusionchart(confusion_matrix);
title('Confusion Matrix ');

%% get accuracy
accuracy=sum(diag(confusion_matrix),"all")/sum(confusion_matrix,"all");
% accuracy = 95.83%
