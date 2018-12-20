function g = readrawb(filename)
    fid = fopen(filename);
    % 连续读取181*217*181个数据，这时候temp是一个长度为181*217*181的向量。
    % 先将rawb中的所有数据传递给temp数组，然后将tempreshape成图片集。
    temp = fread(fid,181*217*26);
    % 所以把它变成了一个181*217行，181列的数组，按照它的代码，这就是181张图片的数据，每一列对应一张图。
    % 生成图片集数组。图片集images数组中每一列表示一张图片。
    images = reshape(temp, 181 , 217, 26);   
    % 读取数组中的第num行，得到数组再reshape成图片原来的行数和列数：181*217。
%     image = images(:, num);
%     image = reshape(image, 181, 217);
    image=zeros(217,181,26);
    for i=1:26
        image(:,:,i)=imrotate(images(:,:,i),-90);
    end
    g = image;
    fclose(fid);
end