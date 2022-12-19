'''
코드 최초 작성자 : 안동대학교 20181107 김주현
ANU 20181107 KIM JU HYEON (GitHub: Kim7obu)
Cobyright 2022.Kim7obu.All right reserved.
'''

import sys
from threading import Thread
import cv2
import time
import math
import mediapipe as mp
import numpy as np
import os
from PIL import Image
from os import environ
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5 import uic

#import Train

# pyqt 관련 오류문구 출력 설정 (문구 출력 해제)
def suppress_qt_warnings():
    environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    environ["QT_SCALE_FACTOR"] = "1"

'''
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★     창 설정
'''
# .ui 파일을 여러개 불러오기 위해서 정의한 함수
def resource_path(relative_path):
    base_path = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

# [창 1] UI 파일 불러오기
form = resource_path('GUI_Integrated.ui')
form_class = uic.loadUiType(form)[0]

# [창 2] UI 파일 불러오기
form_second = resource_path('GUI_Integrated_FR.ui')
form_secondwindow = uic.loadUiType(form_second)[0]

# 검출+학습 창, [창 1]
class WindowClass(QMainWindow, form_class):
    # 프로그램 유의사항 문자열
    INFORMATION = str("\n           Made By Kim7obu, ANU CE 20181107\n\n"
                      + "☆☆☆★★★☆☆☆ HOW TO USE ☆☆☆★★★☆☆☆\n\n"
                      + "1). [ID 입력] 칸에 ID 입력 후 [추가하기] 버튼 누르기\n"
                      + "2). 1번과 같은 방법으로 ID 추가하기\n"
                      + "3). 특정 ID의 얼굴을 캡처하고 싶으면, ID 입력 후 [ID 선택하기] 버튼 누르기\n"
                      + "4). [검출 및 사진 저장] 누르기\n"
                      + "5). 캡처한 얼굴들을 [얼굴 학습 시키기] 버튼을 눌러 학습시키기\n"
                      + "6). [얼굴 인식] 버튼을 누르면 새 창으로 넘어가서 인식을 할 수 있음\n\n"
                      + "추가 사항은 기술문서 참조\n"
                      )

    # ID 입력칸에서 얻어온 ID숫자
    face_id = []
    selected_id = None
    ID_Flag = False

    # 파일 중복 검사 플래그. True이면 중복O, False이면 중복X / faces_haar_train 폴더 내 파일 존재여부
    DupCheckFlag = True
    FileExist = None

    # 의존성 문제때문에 밖으로 빼낸 VideoCapture
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # 디자이너에서 만든 버튼 연결
        self.capture.clicked.connect(self.Capture_Start)  # 스레드 1 / 카메라 열기 및 사진 저장 / 스레드로 넘김
        self.dataset.clicked.connect(self.Make_Dataset)  # 스레드 2 / 사진을 데이터셋으로 만듬 / 스레드로 넘김

        self.CHECK.clicked.connect(self.Check_Notice)  # notice 확인하기
        self.FORCED_EXIT.clicked.connect(self.SHUTDOWN_PROCESS)  # 강제종료 버튼

        self.confirm_ID.clicked.connect(self.getID_Num)  # 사람 ID번호 얻어오기
        self.delete_ID.clicked.connect(self.deleteID_Num)  # ID 입력창 초기화
        self.select_ID.clicked.connect(self.ID_select)  # 학습할 ID 선택하기

        self.Import_ID_List.clicked.connect(self.saveID_List)  # 입력된 ID 텍스트파일로 import
        self.get_ID_List.clicked.connect(self.getID_List)  # 텍스트파일 ID face_id에 넣기
        self.reg_ID_List.clicked.connect(self.showIDList)  # ID리스트 보이기
        self.clear_ID_List.clicked.connect(self.clsIDList) # ID리스트 창 클리어

        self.FeedLabel.setPixmap(QPixmap("wait_pic.jpg"))
        self.FeedLabel.setAlignment(Qt.AlignCenter)

        # 버튼 스타일
        self.capture.setStyleSheet("color: #9966FF;"  # 스레드1
                                       "border-style: solid;"
                                       "border-width: 2px;"
                                       "border-color: #9966FF;"
                                       "border-radius: 3px;"
                                       "font-size: 12px")
        self.dataset.setStyleSheet("color: #0033FF;"  # 스레드2
                                   "border-style: solid;"
                                   "border-width: 2px;"
                                   "border-color: #0033FF;"
                                   "border-radius: 3px;"
                                   "font-size: 12px")
        self.FaceRecogStart.setStyleSheet("color: #006600;"  # 얼굴인식 새창
                                   "border-style: solid;"
                                   "border-width: 2px;"
                                   "border-color: #006600;"
                                   "border-radius: 3px;"
                                   "font-size: 12px")
        self.FORCED_EXIT.setStyleSheet("color: red;"  # 강제종료
                                       "border-style: solid;"
                                       "border-width: 2px;"
                                       "border-color: #FA8072;"
                                       "border-radius: 3px;"
                                       "font-size: 20px")
        self.CHECK.setStyleSheet("color: orange;"  # 주의사항 확인
                                 "border-style: solid;"
                                 "border-width: 2px;"
                                 "border-color: #FFC000;"
                                 "border-radius: 3px;"
                                 "font-size: 18px")

        QMessageBox.information(self, "프로그램 실행 전 주의사항", self.INFORMATION)

        self.textBrowser.setText("Made By Kim7obu, ANU CE 20181107\n" + "주의사항 버튼은 강제종료 버튼 우측에 있습니다.")

    # 여기에 시그널-슬롯 연결 설정 및 함수 설정.
    def main_to_second(self):
        self.hide()  # 메인윈도우 숨김
        self.second = secondwindow()  # 두번째 창
        self.second.exec()  # 두번째 창을 닫을 때 까지 기다림
        self.show()  # 두번째 창을 닫으면 다시 첫 번째 창이 보여짐

    # 프로그램 유의사항
    def Check_Notice(self):
        QMessageBox.information(self, "프로그램 실행 전 주의사항", self.INFORMATION)

    # 강제종료
    def SHUTDOWN_PROCESS(self):
        self.textBrowser.setText("")
        self.textBrowser.setText("---------------------------------------------" + "\n" +
                                 "★☆★ FORCED SHOTDOWN PROCESS INITIATED ★☆★" + "\n" +
                                 "---------------------------------------------")
        print("---------------------------------------------" + "\n" +
              "★☆★ FORCED SHOTDOWN PROCESS INITIATED ★☆★" + "\n" +
              "---------------------------------------------")
        time.sleep(1)
        sys.exit(0)

    # 스레드 1 검사
    def Capture_Start(self):
        self.textBrowser.clear()
        print("clear end")
        self.FileNameException()
        print("exception end")
        if self.DupCheckFlag:  # 중복이면
            print("\n[파일 이름 중복 검사] 중복된 ID 얼굴 사진이 있거나 '해당 ID 선택하기'를 누르셔야 합니다.")
            self.textBrowser.append("\n[파일 이름 중복 검사] 중복된 ID 얼굴 사진이 있거나 '해당 ID 선택하기'를 누르셔야 합니다.")
        elif not self.ID_Flag:
            QMessageBox.warning(self, "경고", "적절한 ID를 선택한 후에 실행해주세요")
        else:
            self.RealWork()
            print("toss to realwork")

    # 중복 ID파일 검사
    def FileNameException(self):

        # 배열 선형탐색. value가 array[]에 존재하면 True, 아니면 False 반환.
        def linear_search(value, array):
            for i, item in enumerate(array):
                if item == value:
                    return True
            return False

        # 폴더 내 파일에서 ID를 빼내서 array에 저장하는 역할임.
        # array = 폴더 안에 있는 사진들의 ID
        array = []
        path = 'faces_haar_train/'
        imagePaths = [os.path.join(path, file) for file in os.listdir(path)]
        # [User].[face_id 인덱스].[이름].[사진 번호].[jpg]
        #    0           1         2        3      4
        for imagePath in imagePaths:
            idNum = int(os.path.split(imagePath)[-1].split(".")[3])
            # 어차피 얼굴 파일은 0~299까지 생성이 되니까, 299번이 캡처가 완료된 마지막 사진일꺼아님?
            if idNum == 299:
                # [사진 번호]가 0인 [이름]을 array에 저장함
                id = os.path.split(imagePath)[-1].split(".")[2]
                print("array에 들어가게 될 id: " + id)
                array.append(id)
        print("FileNameException의 array 값: ", array)
        print("face_id의 길이: ", len(self.face_id))
        print("[창 1]의 face_id 값: ", self.face_id)

        # array의 len이 0이다 = 파일이 없다.
        if len(array) == 0:
            print("\n[ID 중복 검사기] 해당 ID의 사진이 없습니다.")
            self.textBrowser.setText("\n[ID 중복 검사기] 해당 ID의 사진이 없습니다.")
            self.DupCheckFlag = False
            self.FileExist = False
        elif len(array) != 0:  # 만약 파일이 있을 때,
            self.FileExist = True
            print("array의 크기 ",len(array))
            print("face_id의 크기 ",len(self.face_id))

            maximum = len(self.face_id)
            for i in range(maximum):
                try:
                    # 선형 탐색(값, 비교하고자 하는 배열)
                    check1 = linear_search(array[i], self.face_id)
                    check2 = linear_search(self.selected_id, array)
                    print("체크1(array 값과 face_id의 비교) ", check1)
                    print("체크2(사용자 선택 ID와 array의 비교) ", check2)
                except IndexError:
                    # array의 크기보다 face_id의 크기가 더 크다면
                    maximum = len(array)
                    break

                if check1 == True and check2 == True:
                    print("\n[ID 중복 검사기] 중복된 ID 얼굴 사진이 탐지되었습니다.")
                    self.textBrowser.setText("\n[ID 중복 검사기] 중복된 ID 얼굴 사진이 탐지되었습니다.")
                    self.DupCheckFlag = True
                    break
                    array.clear()
                else:
                    self.DupCheckFlag = False
                    array.clear()

    # 스레드 1 시작
    def RealWork(self):
        worker1 = CaptureWorker(self)
        print("set worker1")
        worker1.start()
        print("worker1 start")

    # 스레드 2 시작
    def Make_Dataset(self):
        self.FileNameException()
        worker2 = DatasetWorker(self)
        worker2.start()

    # ID 이름 얻어오기
    def getID_Num(self):
        self.textBrowser.clear()
        ID = self.Person_ID.text()
        if ID == "": # 입력이 없거나
            print("\n[ID 입력] 입력 없음")
            self.textBrowser.append("\n[ID 입력] 입력 없음")
        elif ID.isdigit() == True: # 숫자거나
            print("\n[ID 입력] ID를 문자열로 입력해주세요. (ex. KimJuHyeon)")
            self.textBrowser.setText("\n[ID 입력] ID를 문자열로 입력해주세요. (ex. KimJuHyeon)")
            self.Person_ID.setText("")
        else:  # 문자열일때만 추가함
            self.face_id.append(ID)  # 배열에 값 추가
            WindowClass.ID_Flag = True
            outputString = "\n[ID 입력] ID 이름 ' " + ID + " ' 이 추가되었습니다."
            print(outputString)
            self.Person_ID.setText("")
            self.textBrowser.setText(outputString)
            self.ID_List.setText("")
            for i in range(len(self.face_id)):
                print(self.face_id[i])
                self.ID_List.append("[ID 리스트] " + str(i + 1) + " " + self.face_id[i])

    # GUI 창에 있는 'ID번호' 칸 내용 지우기 및 'ID List'에 값 지우기
    def deleteID_Num(self):
        def linear_search(value, array):
            for i, item in enumerate(array):
                if item == value:
                    return True
            return False

        ids = self.face_id
        self.textBrowser.setText("")
        ID = self.Person_ID.text()

        search = linear_search(ID, self.face_id)
        if len(ids) == 0:
            print("no face id")
            self.Person_ID.setText("")
            self.textBrowser.setText("\n[ID 리스트] ID가 비어있습니다. 'ID 입력'에다가 입력 후 눌러주세요.")
        elif ID == "":
            # ID 입력칸에 아무것도 안적혀있으면
            print("\n[ID 리스트] ID가 선택되지 않았습니다.")
            self.textBrowser.setText("\n[ID 리스트] ID가 선택되지 않았습니다.")
        elif search:
            # ID 입력칸에 있는 사람의 이름을 지울 때
            ids.remove(ID)
            print(ids)
            print("\n[ID 입력] " + ID + " 가 삭제되었습니다.")
            self.textBrowser.append("\n[ID 입력] " + ID + " 삭제되었습니다.")
            self.Person_ID.setText("")
            self.ID_List.setText("")
            for i in range(len(self.face_id)):
                print(self.face_id[i])
                self.ID_List.append("[ID 리스트] " + str(i + 1) + " " + self.face_id[i])
        else:
            # 적힌 ID가 등록된 ID가 아니라면
            print("\n[ID 리스트] 'ID 리스트'에 등록되지 않은 ID입니다.")
            self.textBrowser.setText("\n[ID 리스트] 'ID 리스트'에 등록되지 않은 ID입니다.")

    # ID창에 입력된 ID와, face_id에 있는 놈이 같을때, 입력된 ID를 selected_id에 넣고 문구 출력
    # selected_id가 학습에 들어가게댐
    def ID_select(self):
        self.textBrowser.setText("")
        
        # 배열 선형 탐색. 중복값이 있으면 True, 없으면 False 출력
        def linear_search(value, array):
            for i, item in enumerate(array):
                if item == value:
                    return True
            return False

        # ID = value, face_id = array
        ID = self.Person_ID.text()

        WindowClass.selected_id = None
        check = linear_search(ID, self.face_id)

        if check:
            self.selected_id = ID
            self.textBrowser.clear()
            self.textBrowser.append("\n[ID 선택기] " + ID + " 가 선택되었습니다.")
            print("\n[ID 선택기] " + self.selected_id + " 가 선택되었습니다.")
            self.ID_Flag = True
            print("Id Flag set")
            CaptureWorker.ThRun = True
            print("captureworker flag set")
        else:
            if ID == "":
                self.ID_Flag = False
                self.textBrowser.setText("\n[ID 선택기] 아무것도 선택되지 않았습니다. 선택 후 눌러주세요.")
                print("\n[ID 선택기] 아무것도 선택되지 않았습니다. 선택 후 눌러주세요.")
            else:
                self.ID_Flag = False
                self.textBrowser.setText("\n[ID 선택기] 잘못된 ID가 선택되었습니다. 'ID 리스트' 안에 있는 ID를 선택 후 눌러주세요.")
                print("\n[ID 선택기] 잘못된 ID가 선택되었습니다. 'ID 리스트' 안에 있는 ID를 선택 후 눌러주세요.")

    # ID 리스트를 파일로 저장
    def saveID_List(self):
        if len(self.face_id) == 0:
            self.textBrowser.setText("\n[FILE I/O] ID 리스트가 비었습니다. 'ID 입력'에다가 입력 후 ID를 추가해주세요.")
            print("\n[FILE I/O] ID 리스트가 비었습니다. 'ID 입력'에다가 입력 후 ID를 추가해주세요.")
        else:
            with open('Names.txt', 'w', encoding='UTF-8') as f:
                for name in self.face_id:
                    f.write(name + '\n')
                self.textBrowser.setText("\n[FILE I/O] ID 리스트를 생성하여 내보냈습니다. (파일 이름: Names.txt)")
                print("\n[FILE I/O] ID 리스트를 생성하여 내보냈습니다. (파일 이름: Names.txt)")

    # .txt로 적힌 ID 리스트 얻어옴
    def getID_List(self):
        file = 'Names.txt'
        if os.path.isfile(file):
            print("\n[FILE I/O] ID 리스트가 탐지되었습니다.")
            self.textBrowser.setText("\n[FILE I/O] ID 리스트가 탐지되었습니다.")
            with open('Names.txt', encoding='UTF-8') as f:
                self.face_id = (f.read().splitlines())
            self.ID_List.setText("")
            for i in range(len(self.face_id)):
                print(self.face_id[i])
                self.ID_List.append("[ID 리스트] " + str(i + 1) + " " + self.face_id[i])
        else:
            print("\n[FILE I/O] FILE NOT DETECTED")
            self.textBrowser.setText("\n[FILE I/O] FILE NOT DETECTED")

# ID_List에 보이기
    def showIDList(self):
        self.textBrowser.clear()
        # 이전에 출력된 내용 지워줌
        self.ID_List.setText("")
        if len(self.face_id) == 0:
            print("ID List Empty")
            self.textBrowser.setText("\n[ID 리스트] 등록된 ID가 없습니다. 'ID 입력'란에 입력 및 등록 후 눌러주세요.")
        else:
            print("ID List Show")
            self.textBrowser.setText("\n[ID 리스트] ID 리스트가 출력되었습니다.")
            for i in range(len(self.face_id)):
                print(self.face_id[i])
                self.ID_List.append("[ID 리스트] " + str(i + 1) + " " + self.face_id[i])

# ID_List clear screen
    def clsIDList(self):
        self.Person_ID.setText("")
        self.ID_List.setText("")
        self.textBrowser.setText("\n[ID 리스트] 화면이 지워졌습니다.")

# 인식 창, [창 2]
class secondwindow(QDialog,QWidget,form_secondwindow):
    # 스레드 실행여부 판단
    isThreadWork = False

    # 새로운 창 열림 = 정보공유 안됨.
    # face_id는 기존의 역할과 같음
    face_id = []

    # 새 창 = 새 프로세스 취급. [창 2]용 videocapture
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    minW = 0.1 * cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    minH = 0.1 * cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    # .yml 파일 존재여부 검사
    FileExist = None
    # Names 존재여부
    namesExist = False
    
    def __init__(self):
        super(secondwindow,self).__init__()
        self.initUi()
        self.show()

        pixmap = QPixmap("wait_pic.jpg")
        self.FeedLabel_2.setPixmap(pixmap)

        # ID 리스트 관련
        self.reg_ID_List_2.clicked.connect(self.showIDList)
        self.clear_ID_List_2.clicked.connect(self.clsIDList)

        # 스레드 3 시작
        self.B_START.clicked.connect(self.FR_START)

        # 강제종료
        self.FORCED_EXIT_2.clicked.connect(self.SHUTDOWN_PROCESS)  # 강제종료 버튼
        
        # 버튼 스타일
        self.FORCED_EXIT_2.setStyleSheet("color: red;"  # 강제종료
                                       "border-style: solid;"
                                       "border-width: 2px;"
                                       "border-color: #FA8072;"
                                       "border-radius: 3px;"
                                       "font-size: 20px")
        self.B_START.setStyleSheet("color: #FF0099;"  # 시작
                                   "border-style: solid;"
                                   "border-width: 2px;"
                                   "border-color: #FF0099;"
                                   "border-radius: 3px;"
                                   "font-size: 12px")
        self.B_BACK.setStyleSheet("color: #3399FF;"  # 이전 창으로 돌아가기
                                   "border-style: solid;"
                                   "border-width: 2px;"
                                   "border-color: #3399FF;"
                                   "border-radius: 3px;"
                                   "font-size: 12px")

        # ID리스트 검사 후 실행여부 판단
        file = 'Names.txt'
        if os.path.isfile(file):
            print("\n[FILE I/O] ID 리스트가 탐지되었습니다.")
            self.namesExist = True
            self.textBrowser_2.setText("\n[FILE I/O] ID 리스트가 탐지되었습니다.")
            with open('Names.txt', encoding='UTF-8') as f:
                self.face_id = (f.read().splitlines())
            self.ID_List_2.setText("")
            for i in range(len(self.face_id)):
                print(self.face_id[i])
                self.ID_List_2.append("[ID 리스트] " + str(i + 1) + " " + self.face_id[i])
        else:
            print("\n[FILE I/O] FILE NOT DETECTED")
            self.namesExist = False
            self.textBrowser_2.setText("\n[FILE I/O] FILE NOT DETECTED")
        print("second window class array: ")
        print(self.face_id)

    def FR_START(self): # 얼굴인식 스레드 시작
        if self.namesExist:
            self.isThreadWork = True
            self.worker3 = FaceRecog(self)
            self.worker3.start()
        else:
            self.textBrowser_2.setText("\n[스레드] ID 리스트가 없기 때문에 실행을 할 수 없습니다. 이전 창에서 설정 후 실행해주십시오.")
    
    # UI 시작
    def initUi(self):
        self.setupUi(self)

    # 창 전환. PyQT 시그널로 연결되어있음
    def second_to_main(self):
        # 스레드가 진행되고 있으면 멈추고 전환
        if self.isThreadWork:
            self.worker3.stop()
            self.close()
        else:
            self.close()

    # [창 1]의 그것과 완전히 동일함
    def showIDList(self):
        # 이전에 출력된 내용 지워줌
        self.ID_List_2.clear()
        print(self.face_id)
        if len(self.face_id) == 0:
            print("ID List Empty")
            self.textBrowser_2.setText("\n[ID 리스트] 등록된 ID가 없습니다. 이전 창에서 ID 리스트를 가져오십시오.")
        else:
            print("ID List Show")
            self.textBrowser_2.setText("\n[ID 리스트] ID 리스트가 출력되었습니다.")
            for i in range(len(self.face_id)):
                print(self.face_id[i])
                self.ID_List_2.append("[ID 리스트] " + str(i + 1) + " " + self.face_id[i])

    # [창 1]의 그것과 완전히 동일함
    def clsIDList(self):
        self.ID_List_2.clear()
        self.textBrowser_2.setText("\n[ID 리스트] 화면이 지워졌습니다.")

    # [창 1]의 그것과 완전히 동일함
    def SHUTDOWN_PROCESS(self):
        self.textBrowser_2.clear()
        self.textBrowser_2.setText("---------------------------------------------" + "\n" +
                                 "★☆★ FORCED SHOTDOWN PROCESS INITIATED ★☆★" + "\n" +
                                 "---------------------------------------------")
        print("---------------------------------------------" + "\n" +
              "★☆★ FORCED SHOTDOWN PROCESS INITIATED ★☆★" + "\n" +
              "---------------------------------------------")
        time.sleep(1)
        sys.exit(0)

'''
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★     스레드 설정
'''
# 스레드 1. 캡처 후 저장
class CaptureWorker(QThread):
    # 스레드 실행여부 Flag, imgRGB 복사본
    ThRun = True
    camFeed = None
    saverFlag = True

    # 카메라 작동여부 담당하는 runFlag, success 받아오는 변수 camOK
    runFlag = True
    camOK = None

    # 프레임 측정용
    pTime = 0
    cTime = 0

    # 미디어파이프 얼굴검출 솔루션
    mpFaceDetection = mp.solutions.face_detection
    mpDraw = mp.solutions.drawing_utils
    faceDetection = mpFaceDetection.FaceDetection(0.75)

    # 사진 장수 세는 용도의 count
    count = 0
    maxCount = 300

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

    def run(self):
        if self.ThRun:
            if self.parent.ID_Flag:
                print("\n[CAMERA] 웹캠 상태: 검출 및 저장 - 시작")
                # 다음 사람 등록을 위한 플레그 설정
                self.runFlag = True
                print("runflag set")
                th_cam = Thread(target=self.getCamera)  # getCamera에서 작동하도록 넘김
                th_cam.start()
            else:
                print("\n[CAMERA] ID를 입력하지 않았습니다. 입력 후 실행해주세요.")
                self.parent.textBrowser.append(
                    "\n[CAMERA] ID를 입력하지 않았습니다. 입력 후 실행해주세요." + "\n" + "접근 거부됨")
                self.stop()
        else:
            print("\n[CAMERA] 카메라 Thread 작동 불가")
            self.parent.textBrowser.append("\n[CAMERA] 카메라 Thread 작동 불가")

    # 카메라 '실행'
    def getCamera(self):
        print("\n[CAMERA] 카메라 Thread 작동")
        self.runFlag = True
        self.saverFlag = True
        width = self.parent.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.parent.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        currentFace = self.parent.selected_id
        FaceID_Index = self.parent.face_id.index(currentFace)

        os.mkdir(os.path.join('data',str(currentFace)))
        print(os.path.join('data',str(currentFace)))
        while self.runFlag:
            success, img = self.parent.cap.read()
            self.camOK = success
            if success:
                imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.FaceDetector(img, imgRGB)
                self.h, self.w, self.c = img.shape
                qImg = QImage(self.camFeed, self.w, self.h, self.w * self.c, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qImg)
                self.parent.FeedLabel.setPixmap(pixmap)
                self.parent.FeedLabel.resize(int(width), int(height))
                self.isOver300()
            else:
                self.parent.textBrowser.append("\n[CAMERA] !!! 카메라에 연결할 수 없습니다 !!!")
                print("\n[CAMERA] !!! 카메라에 연결할 수 없습니다 !!!")
                self.stop()
                break
            
            
        print("\n[CAMERA] 카메라 Loop문 종료")
        self.parent.textBrowser.append("\n[CAMERA] 카메라 Loop문 종료")

        # 다음 사람 등록을 위한 플레그 설정
        self.runFlag = True
        self.ThRun = True

    # 얼굴 검출
    def FaceDetector(self, img_, imgRGB_):
        results = self.faceDetection.process(imgRGB_)
        if self.camOK:
            # 영상에서 얼굴이 검출된다면
            if results.detections:
                # 영상에서 얼굴이 감지됐을 때
                for id, detection in enumerate(results.detections):
                    bboxC = detection.location_data.relative_bounding_box  # bounding box class
                    ih, iw, ic = imgRGB_.shape
                    x = int(bboxC.xmin * iw)
                    y = int(bboxC.ymin * ih)
                    w = int(bboxC.width * iw)
                    h = int(bboxC.height * ih)
                    bbox = x, y, w, h

                    # 사진 저장 함수
                    self.PictureSaver(imgRGB_, x, y, w, h)

                    # 얼굴 바운딩 박스
                    cv2.rectangle(imgRGB_, bbox, (255, 0, 255), 1)

                    # FPS
                    cTime = time.time()
                    fps = 1 / (cTime - self.pTime)
                    self.pTime = cTime
                    cv2.putText(imgRGB_, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

                    # 카운터
                    status = " picture captured and saved."
                    status_kr = " 장의 사진이 저장되었습니다."
                    left = " picture left."
                    if self.count == 100:
                        print("100" + status)
                        self.parent.textBrowser.append("\n[사진 카운터] 2" + status_kr)
                    elif self.count == 200:
                        print("200" + status)
                        self.parent.textBrowser.append("[사진 카운터] 200" + status_kr)
                    if self.count == 300:
                        print("300" + status)
                        self.parent.textBrowser.append("[사진 카운터] 300" + status_kr)

                    cv2.putText(imgRGB_, "Caputre Initiated. Keep look at the Camera", (200, 23),
                                cv2.FONT_HERSHEY_PLAIN,
                                2, (0, 255, 0), 2)
                    for i in range(1, self.maxCount // 100):
                        if ((self.count // 100) == i):
                            cv2.putText(imgRGB_, str(int(math.sqrt((self.count // 100) * i) * 100)) + status, (200, 50),
                                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 50, 0), 2)
                            cv2.putText(imgRGB_,
                                        str(int(self.maxCount - (math.sqrt((self.count // 100) * i) * 100))) + left,
                                        (200, 80), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                            
                    # putText까지 싹 긁어서 복사한 내용을 camFeed로 넘겨줌
                    # 이 camFeed를 QPixmap으로 변환시켜서 출력해줌
                    self.camFeed = imgRGB_.copy()
            else:
                self.parent.textBrowser.append("\n[얼굴 검출기] 얼굴 검출이 중단되었습니다. [검출 및 저장] 버튼을 다시 눌러 시작해 주십시오.")
                print("\n[얼굴 검출기] 얼굴 검출이 중단되었습니다. [검출 및 저장] 버튼을 다시 눌러 시작해 주십시오.")
                self.saverFlag = False
                print("saverFlag set False")
                self.stop()
                print("thread1 stopped")
        else:
            print("\n[얼굴 검출기] 카메라 상태 - 나쁨")
            self.parent.textBrowser.append("\n[얼굴 검출기] 카메라 상태 - 나쁨")

    # 얼굴 저장
    def PictureSaver(self, img_, x, y, w, h):
        if self.saverFlag:
            # 사용자가 선택한 ID, face_id의 인덱스를 얻어옴
            currentFace = self.parent.selected_id
            FaceID_Index = self.parent.face_id.index(currentFace)

            # 얼굴 갯수 + 사진 저장
            img = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
            img_onlyface = img[(y):(y + h), (x):(x + w)]  # 얼굴만
            img_onlyface = cv2.resize(img_onlyface, dsize=(112,112))
            
            if (self.count % 50 == 0 and self.count > 30):  # 얼굴만
                cv2.imwrite("data/" + str(currentFace) + "/"+ str(self.count) + ".jpg",
                            img_onlyface, params=[cv2.IMWRITE_JPEG_QUALITY, 100])
            self.count += 1

    def stop(self):
        self.ThRun = False
        self.quit()

    # 플래그만 설정하는 stop함수
    def stop2(self):
        print("스레드1 일시정지")
        self.parent.textBrowser.clear()
        self.parent.textBrowser.append("\n스레드1 일시정지")
        self.ThRun = False

    # 300장되면 스레드 탈출 및 대기사진 출력
    def isOver300(self):
        if self.count == 101:
            self.runFlag = False
            pixmap = QPixmap("wait_pic.jpg")
            self.parent.FeedLabel.setPixmap(pixmap)
            self.parent.FeedLabel.setAlignment(Qt.AlignCenter)
            self.parent.textBrowser.append("[사진 카운터] 얼굴 사진 저장이 완료되었습니다. 카메라를 잠시 닫습니다.")
            self.count = 0
            self.stop()

# 스레드 2. 학습
class DatasetWorker(QThread):
    # 스레드 실행여부 Flag
    ThRun = True

    # 저장된 이미지파일이 있는 곳
    path = 'data/'
    detector = cv2.CascadeClassifier("EXCEL_FILES/haarcascade_frontalface_alt.xml")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    Index_name =[]

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

    def labelCount(self):
        figure_data = 'data'
        self.Index_name=[]
        for figure_datas in os.listdir(figure_data):
            self.Index_name.append(figure_datas)
        return self.Index_name

    def run(self):
        self.ThRun = True
        files = os.listdir(self.path)
        howmanyFiles = len(files)
        if self.ThRun:
            if howmanyFiles == 0:
                self.parent.textBrowser.append("\n[학습기] 얼굴 사진 검출 및 저장 후 실행해주세요.")
                self.stop()
                self.ThRun = True
            else:
                self.parent.textBrowser.append("[학습기] 데이터셋 구축 시작됨")
                pixmap = QPixmap("train.jpg")
                self.parent.FeedLabel.setPixmap(pixmap)
                self.parent.FeedLabel.setAlignment(Qt.AlignCenter)
                print("\n[학습기] 얼굴 학습중...")
                self.parent.textBrowser.append("\n[학습기] 등록된 얼굴을 학습하는 중입니다. 잠시만 기다려주세요.")
                print("next step")
                #self.Index_name = self.labelCount()
                #print(self.Index_name)

                #time.sleep(10)

                __import__('Train')

                print("\n[학습기]{0} 개의 얼굴이 학습되었습니다. 프로그램을 종료합니다.".format(len(np.unique(ids))))
                self.parent.textBrowser.append(
                    "\n[학습기] {0} 개의 얼굴이 학습되었습니다. 프로그램을 종료합니다.".format(len(np.unique(ids))))
                pixmap = QPixmap("complete.jpg")
                self.parent.FeedLabel.setPixmap(pixmap)
                self.parent.FeedLabel.setAlignment(Qt.AlignCenter)
                self.stop()

    def stop(self):
        self.ThRun = False
        self.parent.textBrowser.append("[학습기] 데이터셋 구축 종료됨")
        print("[학습기] 데이터셋 구축 종료됨")
        self.quit()

# 스레드 3. 인식
class FaceRecog(QThread):
    # cv2.release 이후에 재연결을 위해 필요한것
    frame_counter = 0

    # 이름 들어가있는 names
    names = []

    # 스레드 실행여부 Flag, imgRGB 복사본
    ThRun = True
    camFeed = None

    # 카메라 작동여부 담당하는 runFlag, success 받아오는 변수 camOK
    runFlag = True
    camOK = None

    # 프레임 측정용
    pTime = 0
    cTime = 0

    # 미디어파이프 얼굴검출 솔루션
    mpFaceDetection = mp.solutions.face_detection
    mpDraw = mp.solutions.drawing_utils
    faceDetection = mpFaceDetection.FaceDetection(0.75)

    # HaarCascade로 훈련된 trainer.yml 파일이 들어가있는 경로를 불러옴 + HaarCascade 파일 불러옴
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    cascadePath = "EXCEL_FILES/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.names = parent.face_id
        print("Thread 3 face id check: ")
        print(self.names)

    # 훈련된 .yml 파일이 있는지 없는지 찾아주는 함수
    def FileFounder(self):
        file = 'trainer/haar_trainer_GUI.yml'
        if os.path.exists(file):
            self.recognizer.read('trainer/haar_trainer_GUI.yml')  # 현재 상태에선 ver1 사용함.
            self.ThRun = True
        else:
            self.parent.textBrowser_2.append("학습된 데이터셋을 찾을 수 없으므로 스레드를 종료합니다")
            self.stop()

    def run(self):
        self.FileFounder()
        if self.ThRun:
            print("runflag set")
            # 다음 사람 등록을 위한 플레그 설정
            self.runFlag = True
            print("thread set")
            self.th_cam = Thread(target=self.getCamera, daemon=True)  # getCamera에서 작동하도록 넘김
            self.th_cam.start()
        else:
            print("\n[카메라] 카메라 Thread 실행을 할 수 없습니다.")
            self.parent.textBrowser_2.append("\n[카메라] 카메라 Thread 실행을 할 수 없습니다.")

    # 카메라 '실행' / 스레드 1의 그것과 똑같음
    def getCamera(self):
        # 카메라 재연결
        if self.frame_counter == self.parent.cap.get(cv2.CAP_PROP_FRAME_COUNT):
            print("카메라 재연결 확인")
            self.parent.textBrowser_2.append("\n[카메라] 카메라 재연결 중입니다. 잠시만 기다려주세요.")
            self.frame_counter = 0
            self.parent.cap = cv2.VideoCapture(0)
            self.parent.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.parent.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.parent.minW = 0.1 * self.parent.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            self.parent.minH = 0.1 * self.parent.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        print("\n[카메라] 카메라 Thread 시작")
        self.runFlag = True
        width = self.parent.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.parent.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        

        

        while self.runFlag:
            success, img = self.parent.cap.read()
            self.camOK = success
            if success:
                imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.FaceDetector(img, imgRGB)
                self.h, self.w, self.c = img.shape
                qImg = QImage(self.camFeed, self.w, self.h, self.w * self.c, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qImg)
                self.parent.FeedLabel_2.setPixmap(pixmap)
                self.parent.FeedLabel_2.resize(int(width), int(height))

            else:
                self.parent.textBrowser_2.append("\n[카메라] !!! 카메라 연결 불가 !!!")
                print("\n[카메라] !!! 카메라 연결 불가 !!!")
                self.stop()
                break
        print("\n[카메라] 카메라 Loop문 종료됨")
        self.parent.textBrowser_2.append("\n[카메라] 카메라 Loop문 종료됨")


    # 얼굴 검출 / 스레드 1의 그것과 비슷한데, haar cascade 관련 설정이 추가되었음.
    def FaceDetector(self, img_, imgRGB_):
        results = self.faceDetection.process(imgRGB_)
        # 하르 그레이스케일
        results_haar = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)

        # 하르카스케이드 관련 설정
        faces = self.faceCascade.detectMultiScale(
            results_haar,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(secondwindow.minW), int(secondwindow.minH))
        )
        # 카메라가 실행 가능이고
        if self.camOK:
            # 영상에서 얼굴이 감지됐을 때
            if results.detections:
                for id, detection in enumerate(results.detections):
                    bboxC = detection.location_data.relative_bounding_box  # bounding box class
                    ih, iw, ic = imgRGB_.shape
                    x = int(bboxC.xmin * iw)
                    y = int(bboxC.ymin * ih)
                    w = int(bboxC.width * iw)
                    h = int(bboxC.height * ih)
                    bbox = x, y, w, h

                    # 얼굴 바운딩 박스
                    cv2.rectangle(imgRGB_, bbox, (255, 0, 255), 1)
                    for (x, y, w, h) in faces:
                        id, confidence = self.recognizer.predict(results_haar[y:y + h, x:x + w])
                        if ((100 - confidence) >= 75):
                            id = self.names[id]
                        elif ((100 - confidence) < 75):
                            id = "unknown"
                        confidence = "  {0}%".format(round(100 - confidence))
                        cv2.putText(imgRGB_, str(id), (x + 2, y), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
                        cv2.putText(imgRGB_, str(confidence), (x + 5, y + h), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 1)

                    # FPS
                    cTime = time.time()
                    fps = 1 / (cTime - self.pTime)
                    self.pTime = cTime
                    cv2.putText(imgRGB_, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

                    self.camFeed = imgRGB_.copy()
        else:
            print("\n[얼굴 검출] 카메라 상태 - 나쁨")
            self.parent.textBrowser.append("\n[얼굴 검출] 카메라 상태 - 나쁨")

    def stop(self):
        self.ThRun = False
        self.parent.cap.release()
        print("cap release")
        print("th_cam stopped")
        self.quit()


    
'''
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★     메인함수
'''
if __name__ == '__main__':
    # 오류문구 출력 해제
    suppress_qt_warnings()
    
    # 실제 실행
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    app.exec_()