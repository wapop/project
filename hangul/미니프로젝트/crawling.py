from bs4 import BeautifulSoup
import requests
import pandas as pd
from urllib.request import urlopen
import os

#### page돌면서 크롤링

def crawling():
    title=[]
    for page in range(1,10):#페이지 돌면서 크롤링
        url = 'http://www.ihangul.kr/main/bbs/board.php?bo_table=m2_1a&page=2&page='+str(page)

        response = requests.get(url=url)
        soup = BeautifulSoup(response.text, 'html.parser')
        link=soup.select('.list-container > div > div > h2 > a')
        for number in range(0,len(link)):
            title.append(link[number].text.strip().replace('(','').replace(')','').split()[0])
            #괄호 삭제 및 공백으로 분리한 후 0 인덱스 데이터 저장
    file_object = open("word.txt", "a+")
    for a in range(0,len(title)):
        file_object.write(title[a]+'\n')
    file_object.close()


def crawling2():
    title=[]
    for page in range(1,9):#페이지 돌면서 크롤링
        url = 'http://www.ihangul.kr/main/bbs/board.php?bo_table=m2_1b&page=2&page='+str(page)

        response = requests.get(url=url)
        soup = BeautifulSoup(response.text, 'html.parser')
        link=soup.select('.list-container > div > div > h2 > a')
        for number in range(0,len(link)):
            title.append(link[number].text.strip().replace('(','').replace(')','').split()[0])
            #괄호 삭제 및 공백으로 분리한 후 0 인덱스 데이터 저장

    file_object = open("word.txt", "a+")
    for a in range(0,len(title)):
        file_object.write(title[a]+'\n')
    file_object.close()


def crawling3():
    title=[]
    for page in range(1,8):#페이지 돌면서 크롤링
        url = 'http://www.ihangul.kr/main/bbs/board.php?bo_table=m2_2a&page=2&page='+str(page)

        response = requests.get(url=url)
        soup = BeautifulSoup(response.text, 'html.parser')
        link=soup.select('.list-container > div > div > h2 > a')
        for number in range(0,len(link)):
            title.append(link[number].text.strip().replace('(','').replace(')','').split()[0])
            # 괄호 삭제 및 공백으로 분리한 후 0 인덱스 데이터 저장

    file_object = open("word.txt", "a+")
    for a in range(0,len(title)):
        file_object.write(title[a]+'\n')
    file_object.close()


def crawling4():
    title=[]
    for page in range(1,7):#페이지 돌면서 크롤링
        url = 'http://www.ihangul.kr/main/bbs/board.php?bo_table=m2_2b&page=2&page='+str(page)

        response = requests.get(url=url)
        soup = BeautifulSoup(response.text, 'html.parser')
        link=soup.select('.list-container > div > div > h2 > a')
        for number in range(0,len(link)):
            title.append(link[number].text.strip().replace('(','').replace(')','').split()[0])
            # 괄호 삭제 및 공백으로 분리한 후 0 인덱스 데이터 저장

    file_object = open("word.txt", "a+")
    for a in range(0,len(title)):
        file_object.write(title[a]+'\n')
    file_object.close()


def crawling5():
    title=[]
    for page in range(1,4):#페이지 돌면서 크롤링
        url = 'http://www.ihangul.kr/main/bbs/board.php?bo_table=m1_2a&page='+str(page)

        response = requests.get(url=url)
        soup = BeautifulSoup(response.text, 'html.parser')
        link=soup.select('.list-container > div > div > h2 > a')
        for number in range(0,len(link)):
            title.append(link[number].text.strip().replace('(','').replace(')','').split()[0])
            # 괄호 삭제 및 공백으로 분리한 후 0 인덱스 데이터 저장

    file_object = open("word.txt", "a+")
    for a in range(0,len(title)):
        file_object.write(title[a]+'\n')
    file_object.close()


def crawling6():
    title=[]
    urls = []
    for page in range(1,4):#페이지 돌면서 크롤링
        url = 'http://www.ihangul.kr/main/bbs/board.php?bo_table=m1_2a&page='+str(page)

        response = requests.get(url=url)
        soup = BeautifulSoup(response.text, 'html.parser')
        link=soup.select('.list-container > div > div > h2') #이미지 이름 데이터
        temp_url = soup.select('.list-container > div > div > div > div > div ') #이미지 url 데이터

        if page==1:
            num = len(link)
        for name in range(0,len(link)):
            title.append(link[name].text.strip())
            with urlopen(temp_url[name].img['src']) as f:
                if os.path.isdir('./img1')==False:
                    os.makedirs('./img1')
                with open('./img1/'+title[name+((page-1)*num)]+'.jpg','wb') as h:
                    img=f.read() #url주소를 통해 이미지 읽는 함수
                    h.write(img) #읽은 이미지를 파일에다가 쓰기

def crawling7():
    title=[]
    urls = []
    for page in range(1,11):#페이지 돌면서 크롤링
        url = 'http://www.ihangul.kr/main/bbs/board.php?bo_table=m1_2b&page='+str(page)

        response = requests.get(url=url)
        soup = BeautifulSoup(response.text, 'html.parser')
        link=soup.select('.list-container > div > div > h2') #이미지 이름 데이터
        temp_url = soup.select('.list-container > div > div > div > div > div ') #이미지 url 데이터
        if page==1:
            num = len(link)
        for name in range(0,len(link)):
            title.append(link[name].text.strip())
            with urlopen(temp_url[name].img['src']) as f:
                if os.path.isdir('./img2')==False:
                    os.makedirs('./img2')
                with open('./img2/'+title[name+((page-1)*num)]+'.jpg','wb') as h:
                    img=f.read() #url주소를 통해 이미지 읽는 함수
                    h.write(img) #읽은 이미지를 파일에다가 쓰기
crawling()
crawling2()
crawling3()
crawling4()
crawling5()
crawling6()
crawling7()