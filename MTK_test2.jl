using ModelingToolkit
using DifferentialEquations
using StochasticDiffEq
using OrdinaryDiffEq
using Distributions
using Statistics
using Symbolics
using Random
using Printf
using Colors
using Plots

@variables t
D = Differential(t)
#include("synaptic_network.jl")

function HH_neuron_wang_excit(;name,E_syn=0.0,G_syn=2,I_in=0,freq=0.0,phase=0,τ=10)
	sts = @variables V(t)=-65.00 n(t)=0.32 m(t)=0.05 h(t)=0.59 Isyn(t)=0.0 G(t)=0 z(t)=0 
	ps = @parameters E_syn=E_syn G_Na = 52 G_K  = 20 G_L = 0.1 E_Na = 55 E_K = -90 E_L = -60 I_in = I_in G_syn = G_syn V_shift = 10 V_range = 35 τ_syn = 10 freq=freq τ₁ = 0.1 τ₂ = τ 
	

 αₙ(v) = 0.01*(v+34)/(1-exp(-(v+34)/10))
 βₙ(v) = 0.125*exp(-(v+44)/80)

	
 αₘ(v) = 0.1*(v+30)/(1-exp(-(v+30)/10))
 βₘ(v) = 4*exp(-(v+55)/18)
	 
 αₕ(v) = 0.07*exp(-(v+44)/20)
 βₕ(v) = 1/(1+exp(-(v+14)/10))	
	
ϕ = 5 
	
G_asymp(v,G_syn) = (G_syn/(1 + exp(-4.394*((v-V_shift)/V_range))))

	eqs = [ 
		   D(V)~-G_Na*m^3*h*(V-E_Na)-G_K*n^4*(V-E_K)-G_L*(V-E_L)+I_in*(sin(t*freq*2*pi/1000+phase)+1)+Isyn, 
	       D(n)~ϕ*(αₙ(V)*(1-n)-βₙ(V)*n), 
	       D(m)~ϕ*(αₘ(V)*(1-m)-βₘ(V)*m), 
	       D(h)~ϕ*(αₕ(V)*(1-h)-βₕ(V)*h),
           D(G)~(-1/τ₂)*G + z,
	       D(z)~(-1/τ₁)*z + G_asymp(V,G_syn),
	      ]
	
	ODESystem(eqs,t,sts,ps;name=name)
end


function synaptic_network(;name, sys=sys, adj_matrix=adj_matrix)
    syn_eqs= [ 0~sys[1].V - sys[1].V]
	        
    for ii = 1:length(sys)
       	
        presyn = findall(x-> x>0, adj_matrix[ii,:])
        wts = adj_matrix[ii,presyn]		
		presyn_nrn = sys[presyn]
        postsyn_nrn = sys[ii]
		    
        if length(presyn)>0
					
		    ind = [i for i = 1:length(presyn)];
	        eq = [0 ~ sum(p-> (presyn_nrn[p].E_syn-postsyn_nrn.V)*presyn_nrn[p].G*wts[p],ind)-postsyn_nrn.Isyn]
            push!(syn_eqs,eq[1])
			
		else
		    eq = [0~postsyn_nrn.Isyn];
		    push!(syn_eqs,eq[1]);
		 
		end
    end
    popfirst!(syn_eqs)
	
    @named synaptic_eqs = ODESystem(syn_eqs,t)
    
    sys_ode = [sys[ii] for ii = 1:length(sys)]

    @named synaptic_network = compose(synaptic_eqs, sys_ode)
    return structural_simplify(synaptic_network)   

end

Nrns=200
simtime = 100.0
        
E_syn=zeros(1,Nrns);	
G_syn= 0.4*ones(1,Nrns);
freq = 5*ones(Nrns);
phase = pi*ones(Nrns);
τ = 5*ones(Nrns);
mat = rand(Nrns,Nrns);
syn = (sign.(mat .-0.99) .+ 1)/2  
for ii = 1:Nrns
  syn[ii,ii]=0;
end
ind=findall(x -> x>0, syn)
for loop = 1:500
    println(loop);
    I_in  = 0.5*rand(1:Nrns,Nrns);
    #rw=rand(1:Nrns-1);
    #acl =rand(rw+1:Nrns);
    ind_rw = rand(1:length(ind))
    
    # nrn_network is an array of ODESystems using function HH_neuron_wang_excit()
      nrn_network=[]
      nrn_network=[]
	   for ii = 1:Nrns
		
            nn = HH_neuron_wang_excit(name=Symbol("nrn$ii"),E_syn=E_syn[ii],G_syn=G_syn[ii],I_in=I_in[ii],freq=freq[ii],phase=phase[ii],τ=τ[ii])
		
            push!(nrn_network,nn)
      end
       
          #this function takes the array of ODESystems nrn_network and connects them using adjacensy matrix syn 
          #creating a larger ODESystem syn_net
      @named syn_net = synaptic_network(sys=nrn_network,adj_matrix=syn)
  
      prob = ODEProblem(syn_net, [], (0, simtime))
      
      sol = solve(prob,alg_hints=[:stiff],ImplicitEM(),saveat = 0.01,reltol=1e-4,abstol=1e-4)
      
    # THIS IS THE LINE WHICH IS CAUSING THE ERROR. IT UPDATES ADJACENCY MATRIX. IF YOU COMMENT IT YOU WON'T SEE THE ERROR  
      syn[ind[ind_rw]] = syn[ind[ind_rw]] + 0.01;
end
