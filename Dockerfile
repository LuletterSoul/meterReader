FROM python:3.6
MAINTAINER XiangdeDe Liu <qq313700046@icloud.com>
VOLUME /tmp
#打包的工作目录
WORKDIR /build/ins
# 将源代码加入到容器中
ADD / /build/ins
#COPY requirements.txt ./
RUN pip install -r requirements.txt
#映射端口
EXPOSE 8080
#加入容器启动命令
CMD ["python","manage.py","runserver","0.0.0.0:8080"]
#ENTRYPOINT ["java","-Djava.security.egd=file:/dev/./uradindom","-jar","/app.jar"]