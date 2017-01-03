% Property of Michael Castanieto
% All Rights Reserved 
% Do Not Copy -- GET YOUR OWN CODE!!!
prpath = 'prtools';
addpath(prpath);

fid = fopen('Skin_NonSkin.txt');
data = textscan(fid,'%f%f%f%f');
fclose(fid);

R = data{1};
G = data{2};
B = data{3};
skinClass = data{4};

sampleLength = length(skinClass);
sampleSkinLength = length(skinClass(skinClass==1));
sampleNonSkinLength = length(skinClass(skinClass==2));

skinR = R(skinClass==1);
skinG = G(skinClass==1);
skinB = B(skinClass==1);
nonSkinR = R(skinClass==2);
nonSkinG = G(skinClass==2);
nonSkinB = B(skinClass==2);

data = [skinR skinG skinB; nonSkinR nonSkinG nonSkinB];
labs = genlab([sampleSkinLength, sampleNonSkinLength]);
z = prdataset(data,labs);
z = setlablist(z, char('Non Skin','Skin'));
z = setfeatlab(z, char('R','G','B'));
z = setprior(z,[sampleSkinLength; sampleNonSkinLength]/sampleLength);
z = setname(z, 'Skin Segmentation');

[x, y] = gendat(z,[round(sampleNonSkinLength*0.80) round(sampleSkinLength*0.80)]);

w1 = qdc(y);
w2 = ldc(y);
w3 = nmsc(y);
