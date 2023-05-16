
#include <unistd.h>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <time.h>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <string>

using namespace std;

char st[256];
long seed;

//mmm, pi
const double pi=4.0*atan(1.0);

//production
//flag
const int q_production_run=1;

//training
const int epoch_report=20;
const int number_of_trajectories=10000;
const int number_of_trajectories_histogram=number_of_trajectories;

//size- and time parameters (fixed)
const double trajectory_time=1;
const double timestep=1e-3;
const int trajectory_steps=trajectory_time/timestep;

//for averaging
double wd[number_of_trajectories];
double he[number_of_trajectories];
double pos[number_of_trajectories];

//plot potential
const int potential_pics=9;

int traj_number=0;
int potential_pic_counter=0;

double potential_plot_increment=5;
double pos_time[number_of_trajectories][potential_pics+2];


//potential parameters
double c1,c2,c4;

//boundary values
double well_depth=5.0;
double c1_boundary=0.0;
double c2_boundary=-2.0*well_depth; //well depth is c2/2
double c4_boundary=-0.5*c2_boundary; //ensures minima at /pm 1

//model parameters
double position;

const int net_step=1;
const int number_of_report_steps=1000;
const int report_step=trajectory_steps/number_of_report_steps;

//net parameters
const int depth=4; //stem of mushroom net, >=1 (so two-layer net at least)
const int width=4;
const int width_final=10;

const int number_of_inputs=2;
const int number_of_outputs=3;
const int number_of_net_parameters=number_of_inputs*width+width*width*(depth-1)+width*width_final+width_final*number_of_outputs+depth*width+width_final+number_of_outputs;

double inputs[number_of_inputs];
double hidden_node[width][depth];
double outputs[number_of_outputs];
double hidden_node_final[width_final];

//aMC hyperparms
int n_scale=100;
int q_layer_norm=1;
double epsilon=0.01;
double sigma_mutate=0.02;
double sigma_mutate_initial=0.0;

int q_ok=0;
int consec_rejections;

long n_reset=0;

double np;
double phi;

//registers
double mutation[number_of_net_parameters];
double mean_mutation[number_of_net_parameters];
double net_parameters[number_of_net_parameters];
double net_parameters_holder[number_of_net_parameters];

int initial_state;
int number_state_one;
double work_time[number_of_report_steps+1][2]; //starting states P,M
double energy_time[number_of_report_steps+1][2];
double position_time[number_of_report_steps+1][2];
double work_time_variance[number_of_report_steps+1][2];
double energy_time_variance[number_of_report_steps+1][2];
double position_time_variance[number_of_report_steps+1][2];

//vars int
int generation=0;
int record_trajectory=1;

double tau;
double heat;
double work;
double energy;
double min_var;
double mean_heat;
double mean_work;
double histo_width;
double tau_physical;
double best_delta_f_est;
double mean_prob_erasure;

//functions void
void ga(void);
void learn(void);
void read_net(void);
void averaging(void);
void store_net(void);
void initialize(void);
void output_net(void);
void mutate_net(void);
void restore_net(void);
void equilibrate(void);
void jobcomplete(int i);
void initialize_net(void);
void run_trajectory(void);
void reset_registers(void);
void reset_potential(void);
void scale_mutations(void);
void output_histogram(void);
void run_net(int step_number);
void output_trajectory(int gen1);
void run_trajectory_average(void);
void langevin_step(int step_number);
void naive_protocol(int step_number);
void record_position(int step_number);
void output_potential(int step_number);
void update_potential(int step_number);
void output_generational_data(int gen1);
void output_trajectory_average_data(void);
void output_trajectory_data(int step_number);
void output_histogram_position(int time_slice);
void record_trajectory_averages(int step_number);
void plot_function(const char *varname, const char *xlabel, double x_min, double x_max, const char *ylabel, double y_min, double y_max);

//functions double
double test_phi(void);
double potential(void);
double gauss_rv(double sigma);

int main(void){
  
//RN generator
initialize();

//aMC
//learn();

//GA
ga();

}

void initialize(void){

//clean up
sprintf(st,"rm report_*");
cout << st << endl;
cout << endl;
system(st);

if(generation>1){
sprintf(st,"rm net_*_gen_%d.dat",generation-2);
cout << st << endl;
cout << endl;
system(st);
}

ifstream infile0("input_parameters.dat", ios::in);
while (!infile0.eof ()){infile0 >> seed >> generation;}

//seed RN generator
srand48(seed);

//initialize net
if(generation==0){initialize_net();}
else{read_net();}

//optimal value
//sprintf(st,"opt.dat");
//ofstream output_opt(st,ios::out);
//output_opt << lambda_final*lambda_final/(trajectory_time + 2.0) << " " << 0 << endl;
//output_opt << lambda_final*lambda_final/(trajectory_time + 2.0) << " " << 1 << endl;
//output_opt.close();

}



double gauss_rv(double sigma){

double r1,r2;
double g1;
double two_pi = 2.0*pi;

r1=drand48();
r2=drand48();

g1=sqrt(-2.0*log(r1))*sigma*cos(two_pi*r2);

return (g1);

}


void update_potential(int step_number){

double e1,e2;

//initial energy
e1=potential();

//naive protocol
//naive_protocol(step_number);

//net protocol
run_net(step_number);

//final time step
if(step_number==trajectory_steps){reset_potential();}

//final energy
e2=potential();

//work
work+=e2-e1;

//log energy
energy=e2;

}

void run_trajectory(void){

int i;

//initial position
equilibrate();

//run traj
for(i=0;i<=trajectory_steps;i++){

//output data
output_trajectory_data(i);

//update potential
if(i!=trajectory_steps){output_potential(i);}
update_potential(i);

//update position
langevin_step(i);

//record position
record_position(i);

//record trajectory averages
record_trajectory_averages(i);

//update time
tau+=1.0/(1.0*trajectory_steps);
tau_physical+=1.0*timestep;

}

//final-time data
output_potential(trajectory_steps);
output_trajectory_data(trajectory_steps);

//increment trajectory counter
traj_number++;


}

void langevin_step(int step_number){

double e1,e2;

//initial energy
e1=potential();

//take step
double grad=c1+2.0*c2*position+4.0*c4*position*position*position;
double a1=-1.0*grad*timestep;
double a2=sqrt(2.0*timestep);
position+=a1+a2*gauss_rv(1.0);

//final energy
e2=potential();

//heat increment
heat+=e2-e1;

}


void output_trajectory_data(int step_number){

if(record_trajectory==1){
if((step_number % report_step==0) || (step_number==trajectory_steps)){

sprintf(st,"report_position_gen_%d.dat",generation);
ofstream out1(st,ios::app);

sprintf(st,"report_work_gen_%d.dat",generation);
ofstream out2(st,ios::app);

sprintf(st,"report_heat_gen_%d.dat",generation);
ofstream out3(st,ios::app);

sprintf(st,"report_c1_gen_%d.dat",generation);
ofstream out4(st,ios::app);

sprintf(st,"report_c2_gen_%d.dat",generation);
ofstream out5(st,ios::app);

sprintf(st,"report_c4_gen_%d.dat",generation);
ofstream out6(st,ios::app);

sprintf(st,"report_energy_gen_%d.dat",generation);
ofstream out7(st,ios::app);

out1 << tau_physical << " " << position << endl;
out2 << tau_physical << " " << work << endl;
out3 << tau_physical << " " << heat << endl;
out4 << tau_physical << " " << c1 << endl;
out5 << tau_physical << " " << c2 << endl;
out6 << tau_physical << " " << c4 << endl;
out7 << tau_physical << " " << potential() << endl;


}}

}


void plot_function(const char *varname, const char *xlabel, double x_min, double x_max, const char *ylabel, double y_min, double y_max){

//sprintf(st,"cp /Users/swhitelam/asymptote/bin/useful_files/crystal_growth_poisoning_simple/config.asy ."); system(st);
//sprintf(st,"cp /Users/swhitelam/asymptote/bin/useful_files/crystal_growth_poisoning_simple/graph_routines.asy ."); system(st);

const char *varname1="cee";
const char *varname1a="c1";
const char *varname1b="c2";
const char *varname1c="c4";

const char *varname2="potential";
const char *varname2b="pos_time";
const char *varname2c="boltz";

const char *varname3="position";
const char *varname3b="energy";
const char *varname3c="work";

const char *varname4a="work_average_state_0";
const char *varname4b="work_average_state_1";
const char *varname4c="energy_average_state_0";
const char *varname4d="energy_average_state_1";
const char *varname4e="position_average_state_0";
const char *varname4f="position_average_state_1";

 //output file
 sprintf(st,"report_%s.asy",varname);
 ofstream output_interface_asy(st,ios::out);
 
 //write output file
 output_interface_asy << "import graph;" << endl;
 output_interface_asy << "import stats;"<< endl;
 
 output_interface_asy << "from \"graph_routines.asy\" access *;"<< endl;
 
 output_interface_asy << "picture p2;"<< endl;
 output_interface_asy << "defaultpen(1.5);"<< endl;
 
 output_interface_asy << "real ymin=" << y_min << ";"<< endl;
 output_interface_asy << "real ymax=" << y_max << ";"<< endl;
 
 output_interface_asy << "real xmin=" << x_min << ";"<< endl;
 output_interface_asy << "real xmax=" << x_max << ";"<< endl;
 
 output_interface_asy << "size(p2,400,400,IgnoreAspect);"<< endl;
 output_interface_asy << "scale(p2,Linear(true),Linear(true));"<< endl;
 
 
 //void simplot_symbol(picture p, string filename,string name,pen pn,int poly,real a1,real a2,real a3,real s1)

if((varname!=varname1) && (varname!=varname2) && (varname!=varname3)){
sprintf(st,"simplot_simple(p2,\"report_%s_gen_%d.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g);",varname,generation,0.0,255.,255.,1.7);
output_interface_asy << st << endl;
}

if(varname==varname1){

sprintf(st,"simplot_simple(p2,\"report_%s_gen_%d.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g);",varname1a,generation,0.1,0.8,0.1,1.2);
output_interface_asy << st << endl;

sprintf(st,"simplot_simple(p2,\"report_%s_gen_%d.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g);",varname1b,generation,0.8,0.1,0.1,1.2);
output_interface_asy << st << endl;

sprintf(st,"simplot_simple(p2,\"report_%s_gen_%d.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g);",varname1c,generation,0.1,0.1,0.8,1.2);
output_interface_asy << st << endl;

}

if(varname==varname2){
for(int i =0;i<potential_pics+2;i++){

sprintf(st,"simplot_simple(p2,\"report_%s_pic_%d_gen_%d.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g);",varname,i,generation,0.1,0.8,0.1,1.2);
output_interface_asy << st << endl;

sprintf(st,"simplot_simple(p2,\"report_%s_pic2_%d_gen_%d.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g);",varname,i,generation,0.1,0.1,0.8,1.1);
output_interface_asy << st << endl;

sprintf(st,"simplot_simple(p2,\"report_%s_%d_gen_%d.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g);",varname2b,i,generation,0.2,0.2,0.8,1.1);
output_interface_asy << st << endl;

sprintf(st,"simplot_simple_dashed(p2,\"report_%s_pic_%d_gen_%d.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g);",varname2c,i,generation,0.0,0.0,0.0,0.5);
output_interface_asy << st << endl;

}
}

if(varname==varname3){

//position
sprintf(st,"simplot_simple(p2,\"report_%s_gen_%d.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g);",varname3,generation,0.0,255.,255.,0.5);
output_interface_asy << st << endl;
sprintf(st,"simplot_simple(p2,\"report_%s_gen_%d.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g);",varname4e,generation,0.0,255.,255.,1.7);
output_interface_asy << st << endl;
sprintf(st,"simplot_simple(p2,\"report_%s_gen_%d.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g);",varname4f,generation,0.0,255.,255.,1.7);
output_interface_asy << st << endl;

//energy
sprintf(st,"simplot_simple(p2,\"report_%s_gen_%d.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g);",varname3b,generation,0.2,0.8,0.2,0.5);
output_interface_asy << st << endl;
sprintf(st,"simplot_simple(p2,\"report_%s_gen_%d.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g);",varname4c,generation,0.2,0.8,0.2,1.7);
output_interface_asy << st << endl;
sprintf(st,"simplot_simple(p2,\"report_%s_gen_%d.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g);",varname4d,generation,0.2,0.8,0.2,1.7);
output_interface_asy << st << endl;

//work
sprintf(st,"simplot_simple(p2,\"report_%s_gen_%d.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g);",varname3c,generation,0.2,0.2,0.8,0.5);
output_interface_asy << st << endl;
sprintf(st,"simplot_simple(p2,\"report_%s_gen_%d.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g);",varname4a,generation,0.2,0.2,0.8,1.7);
output_interface_asy << st << endl;
sprintf(st,"simplot_simple(p2,\"report_%s_gen_%d.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g);",varname4b,generation,0.2,0.2,0.8,1.7);
output_interface_asy << st << endl;

}



 output_interface_asy << "xlimits(p2, xmin, xmax, true);"<< endl;
 output_interface_asy << "ylimits(p2, ymin, ymax, true);"<< endl;
 
 sprintf(st,"%s",xlabel);
 output_interface_asy << "xaxis(p2,Label(\"$"<<st<<"$\",fontsize(30pt)),BottomTop,LeftTicks(Label(fontsize(30)),new real[]{" << x_min <<"," << x_max <<"})); "<< endl;
 sprintf(st,"%s",ylabel);
 output_interface_asy << "yaxis(p2,Label(\"$"<<st<<"$\",fontsize(30pt)),LeftRight,RightTicks(Label(fontsize(30)),new real[]{"<< y_min << "," << y_max <<"})); "<< endl;
 
 
 output_interface_asy << "scale(Linear(true),Linear(true)); "<< endl;
 
 if(varname==varname2){output_interface_asy << "add(p2.fit(600,200),(0,0),S);"<< endl;}
 else{output_interface_asy << "add(p2.fit(250,250),(0,0),S);"<< endl;}

if(q_production_run==0){
sprintf(st,"/usr/local/texlive/2017/bin/x86_64-darwin/asy report_%s.asy",varname);
system(st);

sprintf(st,"open report_%s.eps",varname);
system(st);
}
 
 
}


 

void read_net(void){

int i;

sprintf(st,"net_in_gen_%d.dat",generation);
ifstream infile(st, ios::in);

for(i=0;i<number_of_net_parameters;i++){infile >> net_parameters[i];}

}

void output_net(void){

int i;

//parameter file
sprintf(st,"net_out_gen_%d.dat",generation);
ofstream out_net(st,ios::out);

for(i=0;i<number_of_net_parameters;i++){out_net << net_parameters[i] << " ";}

}


void store_net(void){

int i;
for(i=0;i<number_of_net_parameters;i++){net_parameters_holder[i]=net_parameters[i];}

}

void mutate_net(void){

int i;
for(i=0;i<number_of_net_parameters;i++){mutation[i]=mean_mutation[i]+gauss_rv(sigma_mutate);}
for(i=0;i<number_of_net_parameters;i++){net_parameters[i]+=mutation[i];}

}


void restore_net(void){

int i;
for(i=0;i<number_of_net_parameters;i++){net_parameters[i]=net_parameters_holder[i];}

}

void initialize_net(void){

int i;
for(i=0;i<number_of_net_parameters;i++){net_parameters[i]=gauss_rv(sigma_mutate_initial);}

}

void naive_protocol(int step_number){

//double t1=1.0*step_number/(1.0*trajectory_time);
//tbc

}

void run_net(int step_number){
if(step_number % net_step==0){

int pid=0;

int i,j,k;

double mu=0.0;
double sigma=0.0;
double delta=1e-4;

//inputs
inputs[0]=tau;
inputs[1]=position;

//surface layer
for(i=0;i<width;i++){
hidden_node[i][0]=net_parameters[pid];pid++;
for(j=0;j<number_of_inputs;j++){hidden_node[i][0]+=inputs[j]*net_parameters[pid];pid++;}
}

//layer norm (pre-activation)
if(q_layer_norm==1){
mu=0.0;sigma=0.0;
for(i=0;i<width;i++){mu+=hidden_node[i][0];sigma+=hidden_node[i][0]*hidden_node[i][0];}
mu=mu/width;sigma=sigma/width;
sigma=sqrt(sigma-mu*mu)+delta;
for(i=0;i<width;i++){hidden_node[i][0]=(hidden_node[i][0]-mu)/sigma;}
}

//activation
for(i=0;i<width;i++){hidden_node[i][0]=tanh(hidden_node[i][0]);}


//stem layers
for(j=1;j<depth;j++){
for(i=0;i<width;i++){
hidden_node[i][j]=net_parameters[pid];pid++;
for(k=0;k<width;k++){hidden_node[i][j]+=hidden_node[k][j-1]*net_parameters[pid];pid++;}
}

//layer norm (pre-activation)
if(q_layer_norm==1){
mu=0.0;sigma=0.0;
for(i=0;i<width;i++){mu+=hidden_node[i][j];sigma+=hidden_node[i][j]*hidden_node[i][j];}
mu=mu/width;sigma=sigma/width;
sigma=sqrt(sigma-mu*mu)+delta;
for(i=0;i<width;i++){hidden_node[i][j]=(hidden_node[i][j]-mu)/sigma;}
}

//activation
for(i=0;i<width;i++){hidden_node[i][j]=tanh(hidden_node[i][j]);}

}

//final layer
for(i=0;i<width_final;i++){
hidden_node_final[i]=net_parameters[pid];pid++;
for(j=0;j<width;j++){hidden_node_final[i]+=hidden_node[j][depth-1]*net_parameters[pid];pid++;}
}

//layer norm (pre-activation)
if(q_layer_norm==1){
mu=0.0;sigma=0.0;
for(i=0;i<width_final;i++){mu+=hidden_node_final[i];sigma+=hidden_node_final[i]*hidden_node_final[i];}
mu=mu/width_final;sigma=sigma/width_final;
sigma=sqrt(sigma-mu*mu)+delta;
for(i=0;i<width_final;i++){hidden_node_final[i]=(hidden_node_final[i]-mu)/sigma;}
}

//activation
for(i=0;i<width_final;i++){hidden_node_final[i]=tanh(hidden_node_final[i]);}

//outputs
for(i=0;i<number_of_outputs;i++){
outputs[i]=net_parameters[pid];pid++;
for(j=0;j<width_final;j++){outputs[i]+=hidden_node_final[j]*net_parameters[pid];pid++;}
}

//potential protocol
c1=c1_boundary+outputs[0];
c2=c2_boundary+outputs[1];
c4=c4_boundary+outputs[2];

}}


void learn(void){

//establish initial phi
record_trajectory=0;
run_trajectory_average();
phi=np;

while(2>1){

output_trajectory(generation);
output_generational_data(generation);

store_net();
mutate_net();
run_trajectory_average();

if(np<=phi){q_ok=1;phi=np;}
else{q_ok=0;restore_net();}

if((q_production_run==0) && (q_ok==1)){
cout << generation << " phi = " << np << ", <P>= " << mean_prob_erasure << ", <W>= " << mean_work << ", <Q> = " << mean_heat << endl;
cout << endl;
}

scale_mutations();
generation++;

}
}


void jobcomplete(int i){

 //sprintf(st,"rm jobcomplete.dat");
 //system(st);
 
 sprintf(st,"jobcomplete.dat");
 ofstream output_job(st,ios::out);
 output_job << i << endl;
 output_job.close();

}


void scale_mutations(void){

int i;

double x1=0.0;
double m1=0.0;

if(q_ok==1){

consec_rejections=0;

//mutations
for(i=0;i<number_of_net_parameters;i++){

x1=mutation[i];
m1=mean_mutation[i];

mean_mutation[i]+=epsilon*(x1-m1);

}}
else{

consec_rejections++;

if(consec_rejections>=n_scale){

n_reset++;
consec_rejections=0;
sigma_mutate*=0.95;

for(i=0;i<number_of_net_parameters;i++){mean_mutation[i]=0;}

}}


}


void run_trajectory_average(void){

int i;

reset_registers();

for(i=0;i<number_of_trajectories;i++){

run_trajectory();
wd[i]=work;
he[i]=heat;
pos[i]=position;
}

//averaging (sets phi)
averaging();
output_trajectory_average_data();
traj_number=0;

}


void output_generational_data(int gen1){
if(gen1 % 100==0){

sprintf(st,"report_mean_work.dat");
ofstream out1(st,ios::app);

sprintf(st,"report_mean_heat.dat");
ofstream out2(st,ios::app);

sprintf(st,"report_mean_prob_erasure.dat");
ofstream out3(st,ios::app);

sprintf(st,"report_phi.dat");
ofstream out4(st,ios::app);

out1 << generation << " " << mean_work << endl;
out2 << generation << " " << mean_heat << endl;
out3 << generation << " " << mean_prob_erasure << endl;
out4 << generation << " " << phi << endl;

}}


void output_histogram(void){

sprintf(st,"report_wd_gen_%d.dat",generation);
ofstream output_wd(st,ios::out);

sprintf(st,"report_wd_weighted_gen_%d.dat",generation);
ofstream output_wd_weighted(st,ios::out);

int i;
int nbin=0;
const int bins=50;

double histo[bins];
double maxwork=0.0;
double minwork=0.0;

//work recorded elswehere
for(i=0;i<bins;i++){histo[i]=0.0;}

for(i=0;i<number_of_trajectories_histogram;i++){

if(i==0){maxwork=wd[i];minwork=wd[i];}
else{

if(wd[i]>maxwork){maxwork=wd[i];}
if(wd[i]<minwork){minwork=wd[i];}
}}

//record width
histo_width=(maxwork-minwork)/(1.0*bins);

//safeguard
if((fabs(maxwork)<1e-6) && (fabs(minwork)<1e-6)){

histo_width=0.1;
maxwork=1.0;
minwork=0.0;

}

for(i=0;i<number_of_trajectories_histogram;i++){

nbin=(int) (1.0*bins*(wd[i]-minwork)/(maxwork-minwork));
if(nbin==bins){nbin--;}

histo[nbin]+=1.0/(1.0*number_of_trajectories_histogram);

}

//output
double w1;
for(i=0;i<bins;i++){

if(histo[nbin]>0.5/(1.0*number_of_trajectories_histogram)){
w1=maxwork*i/(1.0*bins)+minwork*(1.0-i/(1.0*bins))+0.5*(maxwork-minwork)/(1.0*bins);

output_wd << w1 << " " << histo[i]/histo_width << endl;
output_wd_weighted << w1 << " " << exp(-w1)*histo[i] << endl;


}
}

//plot histogram
plot_function("wd","W",-maxwork,maxwork,"P(W)",0,5.0/(histo_width*bins));

}


void output_trajectory(int gen1){
if((gen1 % epoch_report==0) && (gen1>0)){

//histograms
run_trajectory_average();
output_histogram();
for(int i=0;i<potential_pics+2;i++){output_histogram_position(i);}


//output trajectory
record_trajectory=1;

run_trajectory();
plot_function("cee","t",-0.25,1.1*trajectory_time,"c",-10,10);
plot_function("position","t",-0.1,1.1*trajectory_time,"x",-well_depth-0.5,4);
plot_function("potential","x",-2.0,potential_plot_increment*(potential_pics+1.5),"U",-well_depth-1,4);

record_trajectory=0;

}}


void ga(void){

record_trajectory=0;

//mutate net
mutate_net();

//calculate order parameter
run_trajectory_average();

//output evolutionary order parameter
sprintf(st,"report_phi_gen_%d.dat",generation);
ofstream output_phi(st,ios::out);
output_phi << np << endl;

//output other order parameters
sprintf(st,"report_order_parameters_gen_%d.dat",generation);
ofstream output_op(st,ios::out);
output_op << mean_prob_erasure << " " << mean_work << " " << mean_heat << endl;

//output histograms
output_histogram();
for(int i=0;i<potential_pics+2;i++){output_histogram_position(i);}

//output net
output_net();

//trajectory data
record_trajectory=1;
run_trajectory();

//jobcomplete
jobcomplete(1);

if(q_production_run==0){

//plots
plot_function("cee","t",-0.1,1.1*trajectory_time,"c",-10,10);
plot_function("position","t",-0.1,1.1*trajectory_time,"x",-well_depth-0.5,2);
plot_function("potential","x",-2.0,potential_plot_increment*(potential_pics+1.5),"U",-well_depth-0.5,4);

}

}

void averaging(void){

int i,j;

//normalization
double n1=1.0/(1.0*number_of_trajectories);
double en[2]={1.0/(1.0*(number_of_trajectories-number_state_one)),1.0/(1.0*number_state_one)};

//reset counters
mean_work=0.0;
mean_heat=0.0;
mean_prob_erasure=0.0;

for(i=0;i<number_of_trajectories;i++){

mean_work+=wd[i]*n1;
mean_heat+=he[i]*n1;
if(pos[i]<0.0){mean_prob_erasure+=n1;}

}

//time-dependent averages
for(i=0;i<number_of_report_steps+1;i++){
for(j=0;j<2;j++){

work_time[i][j]*=en[j];
energy_time[i][j]*=en[j];
position_time[i][j]*=en[j];

work_time_variance[i][j]*=en[j];
energy_time_variance[i][j]*=en[j];
position_time_variance[i][j]*=en[j];

}}

for(i=0;i<number_of_report_steps+1;i++){
for(j=0;j<2;j++){

work_time_variance[i][j]=work_time_variance[i][j]-work_time[i][j]*work_time[i][j];
energy_time_variance[i][j]=energy_time_variance[i][j]-energy_time[i][j]*energy_time[i][j];
position_time_variance[i][j]=position_time_variance[i][j]-position_time[i][j]*position_time[i][j];

}}

//new phi
np=1-mean_prob_erasure+0.05*(mean_work);
//np=1-mean_prob_erasure+(mean_work*0.0001); //for feedback with t_f=10


}

double potential(void){

double q1;
double p1=position;
double p2=p1*p1;
double p4=p2*p2;

//evalulate potential
q1=c1*p1+c2*p2+c4*p4;

return (q1);

}


void reset_potential(void){

c1=c1_boundary;
c2=c2_boundary;
c4=c4_boundary;

}

void equilibrate(void){

int i;

//initial state
position=1.0;initial_state=1;
if(drand48()<0.5){position=-1.0;initial_state=0;}
if(record_trajectory==1){position=1.0;initial_state=1;}

//for averages
number_state_one+=initial_state;

double equilibration_time=1.0;
int equilibration_steps=equilibration_time/timestep;

reset_potential();

for(i=0;i<=equilibration_steps;i++){langevin_step(i);}

//reset counters
tau=0.0;
work=0.0;
heat=0.0;
tau_physical=0.0;
potential_pic_counter=0;

}

void output_potential(int step_number){
if(record_trajectory==1){

int ok=0;
if(step_number % (trajectory_steps/potential_pics) == 0){ok=1;}
if(step_number==0){ok=1;}
if(step_number==trajectory_steps){ok=1;}

if(ok==1){

sprintf(st,"report_potential_pic_%d_gen_%d.dat",potential_pic_counter,generation);
ofstream out_pic(st,ios::out);

sprintf(st,"report_boltz_pic_%d_gen_%d.dat",potential_pic_counter,generation);
ofstream out_boltz(st,ios::out);

sprintf(st,"report_potential_pic2_%d_gen_%d.dat",potential_pic_counter,generation);
ofstream out_pic2(st,ios::out);

int i;
const int n_points=2000;

double e1;
double e_min=0;
double e_values[n_points];

double x1=-1.5;
double x2=1.5;
double zed=0.0;
double delta_x=0.0;


//record position
double position_holder=position;

for(i=0;i<n_points;i++){

position=x1+(x2-x1)*i/(1.0*n_points-1.0);
e1=potential();

out_pic << potential_plot_increment*potential_pic_counter+position << " " << e1 << endl;

}

//boltzmann weight

//log energies; compute minimum
x1=-2.5;x2=2.5;delta_x=(x2-x1)/(1.0*n_points);
for(i=0;i<n_points;i++){

position=x1+(x2-x1)*i/(1.0*n_points-1.0);
e1=potential();

if(i==0){e_min=e1;}
else{if(e1<e_min){e_min=e1;}}

e_values[i]=e1;

}

//calculate Z
for(i=0;i<n_points;i++){

e_values[i]-=e_min;
zed+=delta_x*exp(-e_values[i]);

}

//plot point
for(i=0;i<n_points;i++){

position=x1+(x2-x1)*i/(1.0*n_points-1.0);
if((position>-1.5) && (position<1.5)){
out_boltz << potential_plot_increment*potential_pic_counter+position << " " << exp(-e_values[i])/zed << endl;
}

}



//reset position
position=position_holder;

//label position of bead
e1=potential();
out_pic2 << potential_plot_increment*potential_pic_counter+position << " " << e1+0.02 << endl;
out_pic2 << potential_plot_increment*potential_pic_counter+position << " " << e1-0.02 << endl;

potential_pic_counter++;

}}}



void output_histogram_position(int time_slice){

sprintf(st,"report_pos_time_%d_gen_%d.dat",time_slice,generation);
ofstream output_pos_time(st,ios::out);

int i;
int nbin=0;
const int bins=50;

double histo[bins];
double maxpos=0.0;
double minpos=0.0;

//pos recorded elswehere
for(i=0;i<bins;i++){histo[i]=0.0;}

for(i=0;i<number_of_trajectories_histogram;i++){

if(i==0){maxpos=pos_time[i][time_slice];minpos=pos_time[i][time_slice];}
else{

if(pos_time[i][time_slice]>maxpos){maxpos=pos_time[i][time_slice];}
if(pos_time[i][time_slice]<minpos){minpos=pos_time[i][time_slice];}
}}

//record width
histo_width=(maxpos-minpos)/(1.0*bins);

//safeguard
if((fabs(maxpos)<1e-6) && (fabs(minpos)<1e-6)){

histo_width=0.1;
maxpos=1.0;
minpos=0.0;

}

for(i=0;i<number_of_trajectories_histogram;i++){

nbin=(int) (1.0*bins*(pos_time[i][time_slice]-minpos)/(maxpos-minpos));
if(nbin==bins){nbin--;}

histo[nbin]+=1.0/(1.0*number_of_trajectories_histogram);

}

//output
double x1;
for(i=0;i<bins;i++){

if(histo[nbin]>0.5/(1.0*number_of_trajectories_histogram)){
x1=maxpos*i/(1.0*bins)+minpos*(1.0-i/(1.0*bins))+0.5*(maxpos-minpos)/(1.0*bins);

output_pos_time << x1 + potential_plot_increment*time_slice << " " << histo[i]/histo_width << endl;


}
}

//histogram plotted from position histogram

}


void record_position(int step_number){

int ok=0;
int entry=0;
int dt=trajectory_steps/potential_pics;

if(step_number==0){ok=1;}
if(step_number==trajectory_steps){ok=1;entry=potential_pics+1;}
if(step_number % dt == 0){ok=1;entry=step_number/dt;}


if(ok==1){pos_time[traj_number][entry]=position;}

}


void reset_registers(void){

int i,j;

traj_number=0;
number_state_one=0;

for(i=0;i<number_of_report_steps+1;i++){
for(j=0;j<2;j++){

work_time[i][j]=0.0;
energy_time[i][j]=0.0;
position_time[i][j]=0.0;

work_time_variance[i][j]=0.0;
energy_time_variance[i][j]=0.0;
position_time_variance[i][j]=0.0;

}}

}



void record_trajectory_averages(int step_number){

int s1;

if(step_number % report_step==0){

s1=step_number/report_step;
if(s1<=number_of_report_steps){

work_time[s1][initial_state]+=work;
energy_time[s1][initial_state]+=energy;
position_time[s1][initial_state]+=position;

work_time_variance[s1][initial_state]+=work*work;
energy_time_variance[s1][initial_state]+=energy*energy;
position_time_variance[s1][initial_state]+=position*position;

}}

}

void output_trajectory_average_data(void){

int i;
double t1;

sprintf(st,"report_work_average_state_0_gen_%d.dat",generation);
ofstream out1(st,ios::app);

sprintf(st,"report_work_average_state_1_gen_%d.dat",generation);
ofstream out2(st,ios::app);

sprintf(st,"report_work_variance_state_0_gen_%d.dat",generation);
ofstream out3(st,ios::app);

sprintf(st,"report_work_variance_state_1_gen_%d.dat",generation);
ofstream out4(st,ios::app);

sprintf(st,"report_energy_average_state_0_gen_%d.dat",generation);
ofstream out5(st,ios::app);

sprintf(st,"report_energy_average_state_1_gen_%d.dat",generation);
ofstream out6(st,ios::app);

sprintf(st,"report_energy_variance_state_0_gen_%d.dat",generation);
ofstream out7(st,ios::app);

sprintf(st,"report_energy_variance_state_1_gen_%d.dat",generation);
ofstream out8(st,ios::app);

sprintf(st,"report_position_average_state_0_gen_%d.dat",generation);
ofstream out9(st,ios::app);

sprintf(st,"report_position_average_state_1_gen_%d.dat",generation);
ofstream out10(st,ios::app);

sprintf(st,"report_position_variance_state_0_gen_%d.dat",generation);
ofstream out11(st,ios::app);

sprintf(st,"report_position_variance_state_1_gen_%d.dat",generation);
ofstream out12(st,ios::app);

for(i=0;i<number_of_report_steps+1;i++){

t1=timestep*i*report_step;

out1 << t1 << " " << work_time[i][0] << endl;
out2 << t1 << " " << work_time[i][1] << endl;
out3 << t1 << " " << work_time_variance[i][0] << endl;
out4 << t1 << " " << work_time_variance[i][1] << endl;


out5 << t1 << " " << energy_time[i][0] << endl;
out6 << t1 << " " << energy_time[i][1] << endl;
out7 << t1 << " " << energy_time_variance[i][0] << endl;
out8 << t1 << " " << energy_time_variance[i][1] << endl;

out9 << t1 << " " << position_time[i][0] << endl;
out10 << t1 << " " << position_time[i][1] << endl;
out11 << t1 << " " << position_time_variance[i][0] << endl;
out12 << t1 << " " << position_time_variance[i][1] << endl;

}

}







