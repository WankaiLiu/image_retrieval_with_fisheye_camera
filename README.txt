
以下假设需求场景为3个
一、数据库采集
1、采集数据库时请尽量朝向该场景的特异性大、纹理丰富的场景，避免空旷/无特征的天花板、地板、墙角等；
2、基于1的基础上，建议正向跑一圈、反向跑一圈，即正向反向都采集

二、评估测试
1、准备好3个场景的数据库，并把每个场景及ID（中间隔空格）写在一个list里面，比如database_list.txt：
    /home/xxxxx/database/slamdata801 7
    /home/xxxxx/database/slamdata803 8
    /home/xxxxx/database/slamdata805 9
    
2、准备好测试数据，并把每个场景及ID（中间隔空格）写在一个list里面，比如querytest_list.txt：
    /home/xxxxx/querytest/slamdata801 7
    /home/xxxxx/querytest/slamdata111 8
    /home/xxxxx/querytest/slamdata666 9

3、运行generate_bin,后跟-tlist和上面的querytest_list.txt，将测试数据每10个图像一组、生成为bin文件，比如：
    ./generate_bin -tlist=/home/xxxxx/querytest/querytest_list.txt
   同时，在generate_bin同级目录下，将会生成bin的路径+场景ID的列表query_list.txt

4、运行imdb_sdk_test，后跟-voc=loopC_vocdata.bin、-ptb=loopC_pattern.yml、-dbase=database_list.txt、-tlist=query_list.txt
    如果有些数据库场景不想加入测试，可再在后面继续加不想测试的数据库场景ID。比如：
    ./imdb_sdk_test \
    -voc=/home/xxxxx/config/loopC_vocdata.bin \
    -ptn=/home/xxxxx/config/loopC_pattern.yml \
    -dbase=/home/xxxxx/config/dlist2.txt \
    -tlist=/home/xxxxx/build2/query_list.txt \
    8 5 0 3

三、实时运行
    图像数据将以10个一组bin数据被送入query_list接口，更多具体的用法请依照imdb_sdk.h中的说明进行
