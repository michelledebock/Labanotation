function img = readAndResizeImages(filename)

% read image
im = imread(filename);

% resize image
img = imresize(im,[200 200]);
