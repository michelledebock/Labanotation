clear all;
clear clc;

%% Evaluation document
J = imread('Knipsel.jpg');
sceneImage = rgb2gray(J);

%% For loop
c=cell(15,1);
outputlines=cell(length(c),1);
for i=1:length(c)
    % Read image files  
    c{i}=imread(sprintf('img%d.jpg',i));
    BoxImage = rgb2gray(c{i});
    %figure;
    %imshow(BoxImage);
    hold on;
    boxPoints = detectSURFFeatures(BoxImage);
    %plot(selectStrongest(boxPoints, 1000));
    % detect SURFFeatures
    scenePoints = detectSURFFeatures(sceneImage);
    hold on;
    [boxFeatures, boxPoints] = extractFeatures(BoxImage, boxPoints);
    [sceneFeatures, scenePoints] = extractFeatures(sceneImage, scenePoints);
    boxPairs = matchFeatures(boxFeatures, sceneFeatures,'MaxRatio', 0.9);
    matchedBoxPoints = boxPoints(boxPairs(:, 1), :);
    matchedScenePoints = scenePoints(boxPairs(:, 2), :);
    %figure;
    %showMatchedFeatures(BoxImage, sceneImage, matchedBoxPoints, ...
    %    matchedScenePoints, 'montage');
    %title('Putatively Matched Points (Including Outliers)');
    
    % Step 5: Locate the Object in the Scene Using Putative Matches
    try
        [tform, inlierBoxPoints, inlierScenePoints] = ...
        estimateGeometricTransform(matchedBoxPoints, matchedScenePoints, 'similar');
    %Display the matching point pairs with the outliers removed
    %figure;
    %showMatchedFeatures(BoxImage, sceneImage, inlierBoxPoints, ...
     %   inlierScenePoints, 'montage');
    %title('Matched Points (Inliers Only)');
    
     % Get the bounding polygon of the reference image.
    hold on;
    boxPolygon = [1, 1;...                           % top-left
            size(BoxImage, 2), 1;...                 % top-right
            size(BoxImage, 2), size(BoxImage, 1);... % bottom-right
            1, size(BoxImage, 1);...                 % bottom-left
            1, 1];                   % top-left again to close the polygon
    hold on;
        outputlines{i} = transformPointsForward(tform, boxPolygon);
        disp('Match');

    %newBoxPolygon = transformPointsForward(tform, boxPolygon);
    %line(newBoxPolygon(:, 1), newBoxPolygon(:, 2), 'Color', 'y');
    catch
        disp('No match');
        outputlines{i} = 0;
    end
end

figure();
imshow(sceneImage);
hold on;
colors=cell(7,1);
colors{1} = 'b';
colors{2} = 'g';
colors{3} = 'r';
colors{4} = 'c';
colors{5} = 'm';
colors{6} = 'y';
colors{7} = 'k';

l = 1;
for p=1:length(c)
    if(outputlines{p} ~= 0)
        line(outputlines{p}(:, 1), outputlines{p}(:, 2), 'Color', colors{l});
        l = l + 1;
        if(l>7)
            l = 1;
        end
    end
end
title('Detected Laban symbols');
close(1);

%annotation = sprintf('Confidence = %.1f',colors{p});
%test = insertObjectAnnotation(sceneImage,'rectangle',outlines(i,:),annotation);
% %% Test
% J = imread('trainingsetfotos.jpg'); 
% label_str = cell(3,1);
% conf_val = ['Links' 'Voor' 'Achter'];
% for ii=1:3
%     label_str{ii} = ['Confidence: ' num2str(conf_val(ii),'%0.2s') '%'];
% end
% 
% position = [23 373 60 66;35 185 77 81;77 107 59 26];
%  
% sceneImage = insertObjectAnnotation(J,'rectangle',position,label_str,...
%     'TextBoxOpacity',0.9,'FontSize',18);
% figure
% imshow(sceneImage)
% title('Annotated chips');
