function [M,PointInitial,PointAdiabatic]=EllipseQsgs3d(L,D_1,D_2,cdd)
%%%%%%%%%%%%%%%%%%%%%   �����趨   %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%L=50;
%D_1=0.1;%ҩ�������
a_0=8; %��Բ����
b_0=a_0/2;
c_0=b_0;
%D_2=0.8; %����ռ��
%cdd=0.01; %���������ܶ�
%%%%%%%%%%%%%%%%%%%%%   ��ʼ��    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
total_d=L^3*D_1;%ҩ�������
temp_d=0;   %ҩ�������
number_seed=0;%ҩ���������
theta=0;%��ת�Ƕ�
Angel=[cos(theta) -sin(theta);sin(theta) cos(theta)];%��ת�任����
M=zeros(L,L,L);%ҩ�����״̬����





%%%%%%%%%%%%%%%%%%%%  ������� ����ҩ�����     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
while temp_d<total_d
    %�������ҩ�����
    while 1
        x_temp=ceil(L*rand());
        y_temp=ceil(L*rand());
        z_temp=ceil(L*rand());
        if M(x_temp,y_temp,z_temp)~=1;
            number_seed=number_seed+1;%��ǰҩ�������Ŀ
            break;
        end
    end
    %�ڵ�ǰҩ��������ɹ̶���״��Һ��
    
    %x,y,z����תŷ���Ƿֽ�
    alpha=pi*rand();
    beta=pi*rand();
    gamma=pi*rand();
    Alpha=[1,0,0;0,cos(alpha),-sin(alpha);0,sin(alpha),cos(alpha)];
    Beta=[cos(beta),0,-sin(beta);0,1,0;sin(beta),0,cos(beta);];
    Gamma=[cos(gamma),-sin(gamma),0;sin(gamma),cos(gamma),0;0,0,1];    

    %���ж���
    %a=a_0-(1/2)*a_0*rand();
    %b=b_0-(1/2)*b_0*rand();
    a=a_0;
    b=b_0;
    c=c_0;
    for i=1:L
        for j=1:L
            for k=1:L
            Temp=Gamma*Beta*Alpha*[i-x_temp;j-y_temp;k-z_temp];
               if (Temp(1)^2/a^2)+(Temp(2)^2/b^2)+(Temp(3)^2/c^2)<=1
                M(i,j,k)=1;                
               end
            end
        end
    end
    temp_d=sum(sum(sum(fix(M))));
end


%%%%%%%%%%%%%%%%%%%     ��������   %%%%%%%%%%%%%%%%%%%%%%%%%%%%

M=wex3d(L,D_2,cdd,M);



for i=1:L
    for j=1:L
        for k=1:L
        switch(M(i,j,k))
            case 0
                M(i,j,k)=1;
            case 1
                M(i,j,k)=0;
                
            case 1/2
                M(i,j,k)=1/2;
                
        end
        end
    end
end
s=0;
for i=1:L
    for j=1:L
        for k=1:L
            if M(i,j,k)==0
                s=s+1;
                PointInitial(s,1)=i;
                PointInitial(s,2)=j;
                PointInitial(s,3)=k;
            end
        end
    end
end

s=0;
for i=1:L
    for j=1:L
        for k=1:L
            if M(i,j,k)==1/2
                s=s+1;
                PointAdiabatic(s,1)=i;
                PointAdiabatic(s,2)=j;
                PointAdiabatic(s,3)=k;
            end
        end
    end
end



%{
[x,y,z] = meshgrid(1:l,1:l,1:l); 
xs = 1:1:l; 
ys = 1:1:l; 
zs = 1:1:l; 
M(M==1)=NaN;
M(l,l,l)=1;
h = slice(x,y,z,M,xs,ys,zs,'nearest'); 
%h=surf(x,y,z,arrgrid);
set(h,'FaceColor','interp',... 
    'EdgeColor','none') 
camproj perspective 
box on 
colormap gray 
colorbar 
%}