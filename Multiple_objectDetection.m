%% Step 1: Read Images
% Read the reference image containing the object of interest.
I = imread('Place Normal.jpg');
boxImage = rgb2gray(I);
figure;
imshow(boxImage);
title('Image of a Box');

%% Read the target image containing a cluttered scene.
J = imread('trainingsetfotos.jpg');
sceneImage = rgb2gray(J);
figure; 
imshow(sceneImage);
title('Image of Laban Scene');

%% Step 2: Detect Feature Points
% Detect feature points in both images.
boxPoints = detectSURFFeatures(boxImage);
scenePoints = detectSURFFeatures(sceneImage);

%% Visualize the strongest feature points found in the reference image.
figure; 
imshow(boxImage);
title('100 Strongest Feature Points from Box Image');
hold on;
plot(selectStrongest(boxPoints, 100));

%% Visualize the strongest feature points found in the target image.
figure; 
imshow(sceneImage);
title('500 Strongest Feature Points from Scene Image');
hold on;
plot(selectStrongest(scenePoints, 500));

%% Step 3: Extract Feature Descriptors
% Extract feature descriptors at the interest points in both images.
[boxFeatures, boxPoints] = extractFeatures(boxImage, boxPoints);
[sceneFeatures, scenePoints] = extractFeatures(sceneImage, scenePoints);

%% Step 4: Find Putative Point Matches
% Match the features using their descriptors. 
boxPairs = matchFeatures(boxFeatures, sceneFeatures);

%% Display putatively matched features. 
matchedBoxPoints = boxPoints(boxPairs(:, 1), :);
matchedScenePoints = scenePoints(boxPairs(:, 2), :);
figure;
showMatchedFeatures(boxImage, sceneImage, matchedBoxPoints, ...
    matchedScenePoints, 'montage');
title('Putatively Matched Points (Including Outliers)');

%% Step 5: Locate the Object in the Scene Using Putative Matches
% |estimateGeometricTransform| calculates the transformation relating the
% matched points, while eliminating outliers. This transformation allows us
% to localize the object in the scene.
[tform, inlierBoxPoints, inlierScenePoints] = ...
    estimateGeometricTransform(matchedBoxPoints, matchedScenePoints, 'similarity');

%% Display the matching point pairs with the outliers removed
figure;
showMatchedFeatures(boxImage, sceneImage, inlierBoxPoints, ...
    inlierScenePoints, 'montage');
title('Matched Points (Inliers Only)');

%% Get the bounding polygon of the reference image.
boxPolygon = [1, 1;...                           % top-left
        size(boxImage, 2), 1;...                 % top-right
        size(boxImage, 2), size(boxImage, 1);... % bottom-right
        1, size(boxImage, 1);...                 % bottom-left
        1, 1];                   % top-left again to close the polygon

%% Transform the polygon into the coordinate system of the target image.
% The transformed polygon indicates the location of the object in the
% scene.
newBoxPolygon = transformPointsForward(tform, boxPolygon);    

%% Display the detected object.
figure;
imshow(sceneImage);
hold on;
line(newBoxPolygon(:, 1), newBoxPolygon(:, 2), 'Color', 'y');
% title('Detected Box');

% %% Step 7: Detect Second Object
% % Read an image containing the second object of interest.  
% K = imread('Left Forward Normal.jpg');
% elephantImage = rgb2gray(K);
% figure;
% imshow(elephantImage);
% title('Image of Left Forward');
% 
% %% Detect and visualize feature points.
% elephantPoints = detectSURFFeatures(elephantImage);
% figure;
% imshow(elephantImage);
% hold on;
% plot(selectStrongest(elephantPoints, 100));
% title('100 Strongest Feature Points from Elephant Image');
% 
% %% Extract feature descriptors.
% [elephantFeatures, elephantPoints] = extractFeatures(elephantImage, elephantPoints);
% 
% %% Match Features
% elephantPairs = matchFeatures(elephantFeatures, sceneFeatures, 'MaxRatio', 0.9);
% 
% %% Display putatively matched features.
% matchedElephantPoints = elephantPoints(elephantPairs(:, 1), :);
% matchedScenePoints = scenePoints(elephantPairs(:, 2), :);
% figure;
% showMatchedFeatures(elephantImage, sceneImage, matchedElephantPoints, ...
%     matchedScenePoints, 'montage');
% title('Putatively Matched Points (Including Outliers)');
% 
% %% Estimate Geometric Transformation and Eliminate Outliers
% [tform, inlierElephantPoints, inlierScenePoints] = ...
%     estimateGeometricTransform(matchedElephantPoints, matchedScenePoints, 'similarity');
% figure;
% showMatchedFeatures(elephantImage, sceneImage, inlierElephantPoints, ...
%     inlierScenePoints, 'montage');
% title('Matched Points (Inliers Only)');
% 
% %% Display Both Objects
% elephantPolygon = [1, 1;...                                 % top-left
%         size(elephantImage, 2), 1;...                       % top-right
%         size(elephantImage, 2), size(elephantImage, 1);...  % bottom-right
%         1, size(elephantImage, 1);...                       % bottom-left
%         1,1];                         % top-left again to close the polygon
%  
% newElephantPolygon = transformPointsForward(tform, elephantPolygon);    
% 
% % figure;
% % imshow(sceneImage);
% % hold on;
% % line(newBoxPolygon(:, 1), newBoxPolygon(:, 2), 'Color', 'y');
% % line(newElephantPolygon(:, 1), newElephantPolygon(:, 2), 'Color', 'g');
% % title('Detected Elephant and Box');
% 
% 
% %% Detect Third Object
% % Detect a second object by using the same steps as before.
% % Read an image containing the second object of interest.  
% M = imread('Left High.jpg');
% LeftLowImage = rgb2gray(M);
% figure;
% imshow(LeftLowImage);
% title('Image of Left Low');
% 
% %% Detect and visualize feature points.
% LeftLowPoints = detectSURFFeatures(LeftLowImage);
% figure;
% imshow(LeftLowImage);
% hold on;
% plot(selectStrongest(LeftLowPoints, 50));
% title('100 Strongest Feature Points from Left Low Image');
% 
% %% Extract feature descriptors.
% [LeftLowFeatures, LeftLowPoints] = extractFeatures(LeftLowImage, LeftLowPoints);
% 
% %% Match Features
% LeftLowPairs = matchFeatures(LeftLowFeatures, sceneFeatures, 'MaxRatio', 0.9);
% 
% %% Display putatively matched features.
% matchedLeftLowPoints = LeftLowPoints(LeftLowPairs(:, 1), :);
% matchedScenePoints = scenePoints(LeftLowPairs(:, 2), :);
% figure;
% showMatchedFeatures(LeftLowImage, sceneImage, matchedLeftLowPoints, ...
%     matchedScenePoints, 'montage');
% title('Putatively Matched Points (Including Outliers)');
% 
% %% Estimate Geometric Transformation and Eliminate Outliers
% [tform, inlierLeftLowPoints, inlierScenePoints] = ...
%     estimateGeometricTransform(matchedLeftLowPoints, matchedScenePoints, 'similarity');
% figure;
% showMatchedFeatures(LeftLowImage, sceneImage, inlierLeftLowPoints, ...
%     inlierScenePoints, 'montage');
% title('Matched Points (Inliers Only)');
% 
% %% Display Both Objects
% LefLowPolygon = [1, 1;...                                 % top-left
%         size(LeftLowImage, 2), 1;...                       % top-right
%         size(LeftLowImage, 2), size(LeftLowImage, 1);...  % bottom-right
%         1, size(LeftLowImage, 1);...                       % bottom-left
%         1,1];                         % top-left again to close the polygon
%  
% newLeftLowPolygon = transformPointsForward(tform, LefLowPolygon);    
% 
% figure;
% imshow(sceneImage);
% hold on;
% line(newBoxPolygon(:, 1), newBoxPolygon(:, 2), 'Color', 'y');
% line(newElephantPolygon(:, 1), newElephantPolygon(:, 2), 'Color', 'b');
% line(newLeftLowPolygon(:,1),newLeftLowPolygon(:,2), 'Color', 'g'); 
% title('Detected Laban symbols');
% 
% 
% % %% Annotation
% % imds = imageDatastore(' ');
% % numImages = length(evaluationData);
% % result(numImages,:) = struct('Boxes',[],'Scores',[]);
% 
% 
% % %% For loop
% % for i = 1:numImages
% %     
% %     % Read Image
% %     I = readimage(imds,i); 
% %     
% %     % Detect the object of interest
% %     [bboxes, scores] = detect(detector,I,'Threshold',1);
% %     
% %     % Store result 
% %     result(i).Boxes = bboxes;
% %     result(i).Scores = scores;
% % end
% % 
% % % Convert structure to table
% % results = struct2table(result);
% % 
