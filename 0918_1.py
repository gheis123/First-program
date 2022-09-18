#기타 그래프 이론
#서로소 집합: 공통원소가 없는 두 집합(Disjoint Sets)
#합집합: 두개의 원소가 포함된 집합을 하나의 집합으로 합침.
#찾기: 특정한 원소가 속한 집합이 어떤 집합인지 알려주는 연산
#서로소 집합 자료구조는 합치기 찾기 자료구조라고 불리기도 한다.

#A와 B의 루트 노드 A',B'을 각각 찾는다
#A'를 B'의 부모 노드로 설정한다.
#모든 합집합 연산 처리할때까지 반복한다.

#1 2 3 4 5 6 (노드번호)
#1 2 3 4 5 6 (부모)
#Union(1,4) Union(2,3) Union(2,4) Union(5,6)
#1 2 3 1 5 6
#1 2 2 1 5 6
#1 1 2 1 5 6
#1 1 2 1 5 5

#서로소 집합 자료구조에서는 연결성을 통해 손십게 집합의 형태를 확인 가능
#1 2 3 4 // 5 6

#기본적인 형태의 서로소 집합 자료구조에서는 루트 노드에 즉시 접근 불가
#루트 노드를 찾기 위해 부모 테이블을 계속해서 확인하며 거슬러가야함.

#특정한 원소가 속한 집합을 찾기
def find_parent(parent,x):
    #root node를 찾을때까지 재귀 호출
    if parent[x]!=x:
        return find_parent(parent,parent[x])
    return x

#두 원소가 속한 집합을 합치기
def union_parent(parent,a,b):
    a=find_parent(parent,a)
    b=find_parent(parent,b)
    if a<b:
        parent[b]=a
    else:
        parent[a]=b

#노드의 개수와 간선(Union 연산)의 개수 입력받기
v,e=map(int,input().split())
parent=[0]*(v+1) # 부모 테이블 초기화하기

#부모 테이블상에서, 부모를 자기자신으로 초기화
for i in range(1,v+1):
    parent[i]=i

#Union 연산을 각각 수행
for i in range(e):
    a,b=map(int,input().split())
    union_parent(parent,a,b)

#각 원소가 속한 집합 출력하기
print("각 원소가 속한 집합: ",end='')
for i in range(1,v+1):
    print(find_parent(parent,i),end=' ')

print()

#부모 테이블 내용 출력하기
print('부모 테이블: ',end='')
for i in range(1,v+1):
    print(parent[i],end=' ')

#합집합 연산이 편향되게 이뤄지는 경우 찾기(Find)함수가 비효율적
#최악의 경우 모든 노드를 다 확이하게 되어 O(V)

#Union(4,5),Union(3,4),Union(2,3),Union(1,2)

#경로 압축
def find_parent(parent,x):
    #루트 노드가 아니라면, 루트노드를 찾을때까지 재귀적 호출
    if parent[x]!=x:
        parent[x]=find_parent(parent,parent[x])
    return parent[x]

#찾기 함수 호출 이후 해당 노드의 루트 노드가 바로 부모 노드가 된다.
#모든 합집합 함수를 처리후, 각 원소에 대해 찾기 함수를 수행하면
#바로 부모 테이블이 갱신된다.



#서로소 집합을 활용한 사이클 판별
#무방향 그래프 내에서의 사이클 판별시 사용
#각선을 하나씩 확인하며 두 노드의 루트 노드를 확인한다
#1. 루트 노드가 서로 다르면 두 노드에 대해 합집합 연산 수행
#2. 루트 노드가 서로 같으면 사이클이 발생한 것
#그래프에 포함되어 있는 모든 간선에 대해 1번 과정을 반복한다.

#초기단계: 모든 노드에 대해 자기 자신을 부모로 설정하는 형태로 부모 테이블 초기화

#서로소 집합을 활용한 사이클 판별
#특정 원소가 속한 집합을 찾기
def find_parent(parent,x):
    #루트 노드를 찾을때까지 재귀 호출
    if parent[x]!=x:
        parent[x]=find_parent(parent,parent[x])
    return parent[x]

#두 원소가 속한 집합을 합치기
def union_parent(parent,a,b):
    a=find_parent(parent,a)
    b=find_parent(parent,b)
    if a<b:
        parent[b]=a
    else:
        parent[a]=b

#노드의 개수와 간선의 개수 입력받기
v,e=map(int,input().split())
parent=[0]*(v+1) #부모 테이블 초기화

#부모 테이블 상에서, 부모를 자기 자신으로 초기화
for i in range(1,v+1):
    parent[i]=i

cycle=False #사이클 발생 여부

for i in range(e):
    a,b=map(int,input().split())
    #사이클이 발생한 경우 종료
    if find_parent(parent,a)==find_parent(parent,b):
        cycle=True
        break
    else: #사이클이 발생하지 않았다면 합집합 연산 수행
        union_parent(parent,a,b)
if cycle:
    print("사이클 발생했다.")
else:
    print("사이클 발생하지 않았다.")


#크루스칼 알고리즘.
#신장트리: 그래프에서 모든 노드를 포함하면서 사이클이 존재하지 않는 부분 그래프
#모든 노드가 포함되어 서로 연결되면서 사이클이 존재하지 않는다는
#조건은 트리의 조건이기도 하다.
#가능한 신장트리 예시와 신장트리가 아닌 부분 그래프 예시는 서로 다름.

#예를 들어 N개의 도시가 존재하는 상황에서 두 도시 사이에 도로를 놓아
#전체 도시가 서로 연결될 수 있게 도로를 설치하는 경우
# ==>두 도시 A,B를 선택했을 때 A에서 b로 이동하는 경로가 반드시 존재하도록

#최소 신창트리 알고리즘(크루스칼 알고리즘)
#그리디 알고리즘
#간선 데이터를 비용에 따라 오름차순 정렬
#간선 하나씩 확인하며 현재의 간선이 사이클을 발생시키는지 확인
#1. 사이클 발생x: 최소 신장 트리에 포함
#2. 사이클 발생O: 최소 신장 트리에 포함하지 않음
#모든 간선에 대해 2가지 과정을 반복한다.

#example of 크루스칼

#특정 원소가 속한 집합을 찾기
def find_parent(parent,x):
    #루트 노드를 찾을 때까지 재귀 호출
    if parent[x]!=x:
        parent[x]=find_parent(parent,parent[x])
    return parent[x]

#두 원소가 속한 집합을 합치기
def union_parent(parent,a,b):
    a=find_parent(parent,a)
    b=find_parent(parent,b)
    if a<b:
        parent[b]=a
    else:
        parent[a]=b

#노드의 개수와 간선의 개수 입력받기
v,e=map(int,input().split())
parent=[]*(v+1) #부모 테이블 초기화

#모든 간선을 담을 리스트와 최종 비용을 담을 변수
edges=[]
result=0

#부모 테이블 상에서, 부모를 자기자신으로 초기화
for i in range(1,v+1):
    parent[i]=i
#모든 간선에 대한 정보 입력받기
for _ in range(e):
    a,b,cost=map(int,input().split())
    #비용 순으로 정렬하기 위해 튜플의 첫번째 원소를 비용으로 설정한다.
    edges.append((cost,a,b))
#간선을 비용순 정렬
edges.sort()

#간선을 하나씩 확인하며
for edge in edges:
    cost,a,b=edge
    #사이클이 발생하지 않는 경우에만 집합에 포함
    if find_parent(parent,a)!=find_parent(parent,b):
        union_parent(parent,a,b)
        result+=cost

print(result)
#ElogE ->가장 많은 시간을 요구하는 곳은 간선 정렬 수행하는 부분!

#위상 정렬: 사이클이 없는 방향그래프의 모든 노드를
#방향성에 거스르지 않도록 순서대로 나열하는 것을 의미한다.
#ex: 선수과목을 고려한 학습 순서 설정.

#진입차수(Indegree): 특정한 노드로 들어오는 간선의 개수
#진출차수(Outdegree): 특정한 노드에서 나가는 간선의 개수
#큐를 이용하는 위상정렬 알고리즘의 동작과정은 다음과 같다
#1. 진입차수가 0인 모든 노드를 큐에 넣는다
#2. 큐가 빌때까지 다음과정 반복
#1) 큐에서 원소를 꺼내 해당 노드에서 나가는 간선을 그래프에서 제거
#2) 새롭게 진입차수가 0이 된 노드를 큐에 넣는다.
#=>결과적으로, 각 노드가 큐에 들어온 순서가 위상 정렬을 수행한 결과와 같음.
#위상정렬을 적용시, 사이클이 없는 방향그래프여야한다(DAG)

#DAG: 순환하지 않는 방향 그래프
#위상정렬에서는 여러 답 존재가능함
#->한단계에서 큐에 새롭게 들어가는 원소가 2개이상인 경우가 있으면!

#모든 원소를 방문전 큐가 비면, 사이클이 존재한다고 판단
#->사이클 포함된 원소중에서 어떠한 원소도 큐에 들어가지 못하므로
#스택을 활용한 dfs를 이용해 위상 정렬을 수행할 수도 있음.


from collections import deque
#노드의 개수와 간선의 개수를 입력받기
v,e=map(int,input().split())
#모든 노드에 대한 진입차수는 0으로 초기화
indegree=[0]*(v+1)
#각 노드에 연결된 간선 정보를 담기 위한 연결리스트 초기화
graph=[[] for i in range(v+1)]

#방향 그래프의 모든 간선 정보를 입력받기
for _ in range(e):
    a,b=map(int,input().split())
    graph[a].append(b) #정점 A에서 B로 이동가능
    #진입 차수를 1 증가
    indegree[b]+=1

#위상 정렬 함수
def topology_sort():
    result=[] #알고리즘 수행 결과를 담을 리스트
    q=deque() # 큐 기능을 위한 deque 라이브러리 사용
    #처음 시작시 진입차수가 0인 노드를 큐에 삽입
    for i in range(1,v+1):
        if indegree[i]==0:
            q.append(i)
        #큐가 빌때까지 반복

        while q:
            #큐에서 원소 꺼내기
            now=q.popleft()
            result.append(now)
            #해당원소와 연결된 노드들의 진입차수에서 1빼기
            for i in graph[now]:
                indegree[i]-=1
                #새롭게 진입차수가 0이 되는 노드를 큐에 삽입
                if indegree[i]==0:
                    q.append(i)
         #위상 정렬을 수행한 결과 출력
        for i in result:
             print(i, end=' ')


topology_sort()

#기타 알고리즘
#1. Prime Number
def is_prime_number(x):
    for i in range(2,x):
        if x%i==0:
            return False
    return True #소수

print(is_prime_number(4))
print(is_prime_number(7)) #시간 복잡도 X

#약수의 성질: 모든 약수가 가운데를 기준으로 곱셈 연산에 대해 대칭적.
#우리는 특정 자연수의 모든 약수를 찾을 때 가운데 약수(제곱근)까지만 확인하면 된다.

#2. 소수 판별: 개선된 version
import math
#소수 판별 함수(2이상의 자연수에 대해)
def is_prime_number(x):
    for i in range(2,int(math.sqrt(x))+1):
        if x%i==0:
            return False
    return True

print(is_prime_number(4))
print(is_prime_number(7)) 
#시간복잡도는 root(N)  =>Very good

#다수의 소수 판별
#특정한 수의 범위 안에 존재하는 모든 소수를 찾아야한다면?
#에라토스테네스의 체 알고리즘.

#2부터 N까지의 모든 자연수를 나열
#남은 수중에서 아직 처리하지 않은 가장 작은수 i를 찾는다
#남은 수 중에서 i의 배수 모두 제거하고 i는 제거하지 않음
#더이상 반복할 수 없을때까지 2번과 3번의 과정을 반복한다.

import math
n=1000 #2부터 1000까지의 모든 수에 대한 소수 판별
#처음엔 모든 수가 소수(True)인것으로 초기화(0과 1제외함)
array=[True for i in range(n+1)]

#에라토스테네스의 체 알고리즘 수행
#2부터 n의 제곱근까지의 모든 수를 확인하며
for i in range(2,int(math.sqrt(n))+1):
    if array[i]==True: #i가 소수인 경우(남은수인 경우)
        #i를 제외한 i의 모든 배수를 지우기
        j=2
        while i*j<=n:
            array[i*j]=False
            j+=1
#모든 소수를 출력한다.
for i in range(2,n+1):
    if array[i]:
        print(i,end=' ')
#NloglogN =>사실상 선형에 가까울 정도로 매우 빠름.
#하지만 각 자연수에 대한 소수 여부를 저장해야 하므로 메모리가 많이 필요
#10억이 소수인지 아닌지 판별하고자 할때, 에라토스테네스의 체 사용가능???


#3. 투포인터: 두개의 점의 위치를 기록하면서 처리하는 알고리즘.
#리스트에 담긴 데이터에 순차적 접근시, 시작점과 끝점 2개의 점으로 접근할 데이터의 범위를 표현할 수 있음.

#N개의 자연수로 구성된 수열이 있다.
#합이 M인 부분 연속 수열의 개수를 구해보시오.
#제한시간 O(N)

#특정한 합을 가지는 부분 연속 수열찾기:
#시작점과 끝점이 첫번째 원소의 인덱스(0)을 가리키도록 한다.
#현재 부분합이 M과 같다면 카운트
#현재 부분합이 M보다 작으면, end를 1 증가시킴
#현재 부분합이 M보다 크거나 같다면, start+1
#모든 경우를 확인할 때까지 2번부터 4번까지의 과정을 반복한다.


n=5 #데이터의 개수
m=5 #찾고자하는 부분합 M
data=[1,2,3,2,5] #전체 수열

count=0
interval_sum=0
end=0

#start를 차례대로 증가시키며 반복
for start in range(n):
    #end를 가능한 만큼 이동시키기
    while interval_sum<m and end<n:
        interval_sum+=data[end]
        end+=1
    #부분합이 m일때 카운트 증가
    if interval_sum==m:
        count+=1
    interval_sum-=data[start]

print(count)

#구간합 문제: 연속적으로 나열된 N개의 수가 있을때 특정 구간의 모든 수를 합하는 문제
#N개의 정수로 구성된 수열이 있음
#M개의 쿼리정보가 주어짐
#각 쿼리는 left와 right로 구성됨
#각 쿼리에 대해 [left,right] 구간에 포함된 데이터들의 합을 출력해야함
#수행시간 제한은 O(N+M)이여야 한다.

#접두사함(Prefix Sum): 배열의 맨 앞부터 특정 위치까지의 합을 미리 구해놓은것
#N개의 수 위치 각각에 대해 접두사 합을 계산하여 P에 저장한다
#매 M개 쿼리 정보를 확인할 때 구간합은 P[right]-P[left-1]이다.

#   10 20 30 40  50
# 0 10 30 60 100 150
#left=1,right=3 =>P[3]-P[0]=60

n=5
data=[10,20,30,40,50]
#접두사합(Prefix Sum) 배열 계산
sum_value=0
prefix_sum=[0]
for i in data:
    sum_value+=i
    prefix_sum.append(sum_value)

#구간 합 계산(세번째 수부터 네번째 수까지)
left=3
right=4
print(prefix_sum[right]-prefix_sum[left-1])


#개발형 코딩테스트[완성도 높은 하나의 프로그램, 모듈을 적절히 조합하는 능력]
#해커톤: 단기간에 아이디어를 제품화하는 프로젝트 이벤트
#서버, 클라이언트, JSON, REST API, .... 반드시 알아야할 개념

#서버와 클라이언트
#클라이언트가 요청[Request]을 보내면 서버가 응답[Response]한다.
#웹 클라이언트: PC, 노트북, 스마트폰
#웹 서버: 워크스테이션

#client=고객
#서버로 요청을 보내고 응답이 도착할때까지 기다림
#서버로부터 응답을 받은 뒤, 서버의 응답을 화면에 출력함
#ex1) 웹 브라우저: 서버로부터 받은 HTML, CSS 코드를 화면에 적절한 형태로 출력
#ex2) 게임앱: 서버로부터 받은 경험치, 친구 귓속말 정보등을 화면에 적절한 형태로 출력

#Server=서비스 제공자
#클라이언트로부터 받은 요청을 처리해 응답을 전송
#ex1) 웹서버: 로그인 요청을 받아 아이디와 비밀번호가 정확한지 검사하고 그 결과를 응답

#HTTP: 웹상에서 데이터를 주고받기 위한 프로토콜을 의미
#보통은 웹문서(HTML)파일을 주고받는데 사용
#모바일 앱 및 게임 개발등에서  특정 형식의 데이터 주고받는 용도로도 사용
#클라이언트는 요청의 목적에 따라 적절한 HTTP 메서드를 이용해 통신을 진행
#GET: 특정 데이터의 조회
#POST: 특정 데이터의 생성
#PUT: 특정 데이터의 수정
#DELETE: 특정 데이터의 삭제를 요청

#
import requests
target="http://google.com"
response=requests.get(url=target)
print(response.text)

#REST의 등장배경
#HTTP는 GET,POST,PUT,DELETE등의 다양 HTTP 메서드를 지원함
#실제로는 서버가 각 메서드의 기본 설명을 따르지 않아도 프로그램 개발 가능
#하지만 저마다 다른 방식으로 개발하면 문제가 되므로 기준이 되는 아키텍처가 필요

#REST는 각 자원에 대해 자원의 상태에 대한 정보를 주고받는 개발방식
#REST 구성요소
#자원(Resource): URL 이용
#행위(Verb): HTTP 메서드를 이용
#표현(Representations): 페이로드를 이용

#URL: httpsL//www.example.com/users
#HTTP Method: POST
#Payload:{"id": :gildong123", "password": "123123"}

#REST API? 
#프로그램이 상호작용하기 위한 인터페이스
#REST API: REST 아키텍처를 따르는 API를 의미
#REST API 호출: REST 방식을 따르고 있는 서버에 특정한 요청을 전송하는 것을 의미함.

#JSON: 데이터를 주고받는데 사용하는 경량의 데이터 형식
import json
#사전 자료형(dict)데이터 선언)
user={
    "id":"gildong123",
    "password":"12312313",
    "age":30,
    "hobby":["football","programming"]
}

#파이썬 변수를 JSON 객체로 변환
json_data=json.dumps(user,indent=4)
print(json_data)

#JSON 객체 파일 저장 예제
with open("user.json","w",encoding="uft-8") as file:
    json_data=json.dump(user,file,indent=4)

#목킹(Mocking)이란 어떠한 기능이 있는 것처럼 흉내내어 구현한 것을 의미
#가상의 REST API 제공 서비스
#REST API 호출 실습해보기

#https://jsonplaceholder.typicode.com/users/1

#HTTP 메서드: GET
{
    "id":1,
    "name":"Charles",
    "username": "hi",
    "email":"Happyycharles@instargram.com"
    #.......
}

#REST API를 호출하여 회원 정보를 처리하는 예제
import requests
#REST API 경로에 접속하여 응당 데이터 받아오기
target="https://jsonplaceholder.typicode.com/users"
response=requests.get(url=target)

#응답 데이터가 JSON 형식이므로 바로 파이썬 객체로 변환
data=response.json()

#모든 사용자(user) 정보를 확인하며 이름 정보만 삽입
name_list=[]
for user in data:
    name_list.append(user['name'])
print(name_list)

