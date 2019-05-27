function [training_set,test_set] = prepareInputFiles(dsObj)

image_location = fileparts(dsObj.Files{1});

imset = imageSet(strcat(image_location,'\..'),'recursive');
[training_set,test_set] = imset.partition(15);
test_set = test_set.partition(15);
end