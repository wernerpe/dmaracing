import cv2 as cv

class viz():
    def __init__(self, modelparams):
        with open(modelparams) as file:
            params = yaml.load(file, Loader= yaml.FullLoader)

        self.w = 2*params['r']
        self.ltot = params['l']
        self.lup = params['lcg']
        self.llow = self.ltot-self.lup

        self.width = 800
        self.height = 500
        self.width_dist = 1.0 #m
        self.height_dist = (self.height*1.0/self.width)*self.width_dist
        img = 255*np.ones((self.height,self.width,3), np.uint8)
        cv.imshow("viz", img)

    def render(self, state, plottime=False, time=0):
        #state = [xcg, ycg, phi]
        xcg,ycg,phi = self.convertstate(state)
        xsensor = state[0]
        #xcg = state[0]#
        #ycg = state[1]
        #phi = state[2]

        img = 255*np.ones((self.height,self.width,3), np.uint8)
        cv.line(img,self.cords(-1,0),self.cords(1,0),(0,0,0),1)
        cv.line(img,self.cords(0,1),self.cords(0,-1),(0,0,0),1)
        cv.line(img,self.cords(0.25,0.05),self.cords(0.25,-0.05),(0,0,255),4)
        cv.line(img,self.cords(-0.25,0.05),self.cords(-0.25,-0.05),(0,0,255),4)
        cv.line(img,self.cords(0.18,0.03),self.cords(0.18,-0.03),(0,0,0),1)
        cv.line(img,self.cords(-0.18,0.03),self.cords(-0.18,-0.03),(0,0,0),1)
        self.plotpend(img, phi, xcg, ycg)
        wsens = 0.02
        tls = self.p2idx(np.array([[xsensor-wsens],[wsens]]))
        trs = self.p2idx(np.array([[xsensor+wsens],[wsens]]))
        brs = self.p2idx(np.array([[xsensor+wsens],[-wsens]]))
        bls = self.p2idx(np.array([[xsensor-wsens],[-wsens]]))
        pts_sensor = np.array([tls,trs,brs,bls]).reshape(-1,1,2)

        cv.fillPoly(img, [pts_sensor], (0,0,0))

        if(plottime):
            cv.putText(img, "time: "+"%0.2f"%time +" [s]", (20, 40), font, 1, (0, 0, 0), 1, cv.LINE_AA)
        cv.imshow("viz", img)
        cv.waitKey(1)

    def convertstate(self,state):
        x = state[0]
        phi = state[1]
        l = state[2]
        xcg = l*np.sin(phi)+x
        ycg = -l*np.cos(phi)
        return xcg, ycg, phi

    def cords(self, x, y):
        idx0 = np.int(x*self.width*1.0/self.width_dist + self.width*1.0/2)
        idx1 = np.int(-y*self.height*1.0/self.height_dist + self.height*1.0/2)
        return (idx0, idx1)

    def p2idx(self, pt):
        idx0, idx1 = self.cords(pt[0,0],pt[1,0])
        return np.array([idx0, idx1])

    def plotpend(self, img, phi, xcg, ycg):
        tr = np.array([self.w/2, self.lup])
        tl = np.array([-self.w/2, self.lup])
        bl = np.array([-self.w/2, -self.llow])
        br = np.array([self.w/2, -self.llow])

        tc = np.array([0, self.ltot*0.05])
        bc = np.array([0, -self.ltot*0.05])

        R = np.matrix([[np.cos(phi), -np.sin(phi)],[np.sin(phi), np.cos(phi)]])

        sh = np.array([[xcg],[ycg]])

        if(np.cos(phi)):
            dist = -ycg/np.cos(phi)
        else:
            dist = 0
        xtp = -np.sin(phi)*dist
        ytp = np.cos(phi)*dist
        rsh = np.array([[xtp+np.cos(phi)*(self.w/2 +0.005)],[ytp+np.sin(phi)*(self.w/2 +0.005)]])
        lsh = np.array([[xtp-np.cos(phi)*(self.w/2 +0.005)],[ytp-np.sin(phi)*(self.w/2 +0.005)]])
        tr = self.p2idx((R@tr).T+sh)
        tl = self.p2idx((R@tl).T+sh)
        br = self.p2idx((R@br).T+sh)
        bl = self.p2idx((R@bl).T+sh)
        ltc = self.p2idx((R@tc).T+sh+lsh)
        lbc = self.p2idx((R@bc).T+sh+lsh)
        rtc = self.p2idx((R@tc).T+sh+rsh)
        rbc = self.p2idx((R@bc).T+sh+rsh)

        pts = np.array([tr,br,bl,tl]).reshape((-1,1,2))
        cv.fillPoly(img, [pts], (180,0,0))
        cv.line(img, (ltc[0],ltc[1]), (lbc[0],lbc[1]), (0,0,0),2)
        cv.line(img, (rtc[0],rtc[1]), (rbc[0],rbc[1]), (0,0,0),2)
        #breakpoint()

    def close(self,):
        cv.destroyAllWindows()
        cv.waitKey(1)