#Queue: First-In-First-Out ==Last-In-Last-Out[BPS USED]
#from collections import deque

#dq=deque()
#dq.append(123)
#dq.append(456)
#while len(dq):
#    print(dq.popleft())

#card game: 카드가 한장 남을 때까지 반복
#example about N=4 
#1234->234->342->42->24->4
from collections import deque

dq=deque()
num=int(input("Input your number:"))
for i in range(1,num+1):
    dq.append(i) #큐에 다 넣어둔 상태

while len(dq)>1:
    dq.popleft()
    dq.append(dq.popleft())

print(dq.popleft())

#Priority Queue
#내부적으로 힙이라는 완전이진트리로 되어있다.
#어떤 값을 넣어도 가장 큰 값 또는 가장 작은 값이 위치하도록 갱신함.
#삽입,삭제 시간 복잡도는 O(logN) : heap sort
#Priority Queue는 멀티스레딩 환경을 고려하여 스레드간 안전한 모듈이지만
#더 빠른 모델을 사용하기 위하여 heapq를 사용한다.

import heapq

h=[]
heapq.heappush(h,123)
heapq.heappush(h,789)
heapq.heappush(h,456)
while len(h):
    print(heapq.heappop(h))
#출력해보면 작은 순서순으로 출력되는 것을 알 수 있다.

#절댓값 힙
#배열에서 정수 x(!=0)를 넣는다.
#절대값이 가장 작은 값을 출력하고, 그 값을 배열에서 제거한다.
#절대값이 가장 작은 값이 여러개이면, 가장 작은 수를 출력하고
#그 값을 배열에서 제거한다.
#프로그램은 처음에 비어있는 배열에서 시작한다.
#example input:18 1 -1 0 0 0 1 1 -1 -1 2 -2 0 0 0 0 0 0 0
#example output:-1 1 0 -1 -1 1 1 -2 2 0

import heapq
import sys,heapq
input=sys.stdin.readline
hq=[]
result=[]
for _ in range(int(input())):
    x=int(input())
    if x!=0:
        heapq.heappush(hq,(abs(x),x))
    if x==0:
        if len(hq):
            result.append(heapq.heappop(hq)[1])
        else:
            result.append(0)

print("The output is {}".format(result))

#map: key-value (key는 중복될 수 없다) 
#삽입,삭제 O(logN)
#unordered map의 내부 구조는 hash로 되어져 있다.
#python의 dictionary는 해시여서 정렬되어 있지 않고 O(1)이다.

#example(best-seller)
#오늘 하루 동안 팔린 책의 제목이 입력으로 들어왔을 때, 가장 많이 팔린
#책의 제목을 출력하는 프로그램을 작성하라
#5 top top top top kimtop ==>top(output)
books=dict()
num=int(input("Input your books numbers:"))
for _ in range(num):
    book_name=input("Input your book name:")
    if book_name in books:
        books[book_name]+=1
    else:
        books[book_name]=1
max_number=max(books.values())
j=[]
for k,v in books.items():
    if max_number==v:
        j.append(k)
j.sort()
print(books)
print(j[0])



#set->hash로 되어있고, 삽입, 삭제 복잡도는 O(1). 
#example program: 4
#A in 
#B in 
#A out
#C in
#==>output: B,C

people=dict()
num=int(input("Input your number:"))
print("{}번의 기록이 존재합니다.".format(num))
for _ in range(num):
    (a,b)=map(str,input().split())
    people[a]=b
for k,v in people.items():
    if v=='in':
        print(k)

#이 문제를 똑같이 set의 관점에서 살펴보자.
import sys
input=sys.stdin.readline
s=set()
for _ in range(int(input())):
    name,el=input().split()
    if el=='in':
        s.add(name)
    elif el=='out':
        if name in s:
            s.remove(name)

for name in sorted(s,reverse=True):
    print(name)

#<________________NEW___________________>

#Brute-force(완전탐색)

#순열(Permutation)
from itertools import permutations
arr=[0,1,2,3]
cnt=0
for i in permutations(arr,2):
    cnt+=1
    print(i)
print("총 개수는 {}입니다.".format(cnt))


#조합(Combination)
from itertools import combinations
arr=[0,1,2,3]
cnt=0 #cnt 초기화
for i in combinations(arr,2):
    cnt+=1
    print(i)
print("총 개수는 {}입니다.".format(cnt))

#example(백설공주와 일곱난쟁이)
#백설공주는 의자 7,접시7,나이프7개를 준비한다.
#난쟁이가 쓰고 다니는 모자에 100보다 작은 양의 정수를 적었다.
#일곱 난쟁이의 모자에 쓰여있는 숫자의 합은 100이다.
#일곱 난쟁이를 찾는 프로그램을 개발하시오.

#제한 조건: 아홉개의 줄에 1보다 크거나 같고 99보다 작거나 같은 자연수 주어짐
#모든 숫자는 서로 다르며, 항상 답이 유일한 경우만 입력으로 주어지게 된다.
#[7,8,10,13,15,19,20,23,25]=>[7,8,10,13,19,20,23]
#[8,6,5,1,37,30,28,22,36]=>[8,6,5,1,30,28,22]
from itertools import combinations
ex1=[7,8,10,13,15,19,20,23,25]
ex2=[8,6,5,1,37,30,28,22,36]

for i in combinations(ex1,7):
    if sum(i)==100:
        print(list(i))
     
for j in combinations(ex2,7):
    if sum(j)==100:
        print(list(j))

#example 
#유레카이론
#Tn=1+2+3+...+n
#(T)={1,3,6,10,15,21,28.....}
#4=T1+T2, 5=T1+T1+T2, 6=T3 or T2+T2
#몇몇 자연수가 정확하게 3개의 삼각수의 합으로 표현될 수 있는지 궁금해졌음.

#삼각수가 다 다를 필요는 없으며, 3개의 삼각수의 합으로 표현될 수 있는지??
#num3: 10 20 1000 ->1 0 1  여기서 1은 표현가능, 0은 표현 불가능.
#단 K의 범위는 3이상 1000이하이다.

T=[n*(n+1)//2 for n in range(1,46)] #45개의 삼각수 만들기.

def judgement(num):
    for i in range(0,45):
        for j in range(i,45):
            for k in range(j,45):
                if T[i]+T[j]+T[k]==num:
                    return 1
    return 0

print(judgement(10),judgement(20),judgement(1000))
