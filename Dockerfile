FROM python:3.6
MAINTAINER XiangdeDe Liu <qq313700046@icloud.com>
VOLUME /tmp
#打包的工作目录
WORKDIR /build/ins
# 将源代码加入到容器中
ADD / /build/ins
#COPY requirements.txt ./
RUN pip install   -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
#映射端口
EXPOSE 8080
#加入容器启动命令
#CMD ["python","./server/manage.py","runserver","--setting=server.settings.prod","0.0.0.0:8080"]
COPY ./entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]
#ENTRYPOINT ["java","-Djava.security.egd=file:/dev/./uradind1m","-jar","/app.jar"]