function build_opf(net; backend, warm=false, objective=:cost, vw=1.0, dscale=1.0, pq_box=true)
    PowerModels.standardize_cost_terms!(net, order=2)
    _rate(br) = haskey(br, "rate_a") ? Float64(br["rate_a"]) : Inf
    _rate_sq(br) = haskey(br, "rate_a") ? Float64(br["rate_a"])^2 : 1e12
    ref = PowerModels.build_ref(net)[:it][:pm][:nw][0]
    arcdict=Dict(a=>k for (k,a) in enumerate(ref[:arcs])); busdict=Dict(k=>i for (i,(k,v)) in enumerate(ref[:bus]))
    gendict=Dict(k=>i for (i,(k,v)) in enumerate(ref[:gen])); branchdict=Dict(k=>i for (i,(k,v)) in enumerate(ref[:branch]))
    data=(
      bus=[begin bl=[ref[:load][l] for l in ref[:bus_loads][k]]; bs=[ref[:shunt][s] for s in ref[:bus_shunts][k]];
           (i=busdict[k],pd=sum(l["pd"] for l in bl;init=0.0),gs=sum(s["gs"] for s in bs;init=0.0),
            qd=sum(l["qd"] for l in bl;init=0.0),bs=sum(s["bs"] for s in bs;init=0.0),
            vmt=Float64(get(v,"vm",1.0)),
            isctrl=(Int(v["bus_type"]) in (2,3))) end for (k,v) in ref[:bus]],
      gen=[(i=gendict[k],cost1=v["cost"][1],cost2=v["cost"][2],cost3=v["cost"][3],bus=busdict[v["gen_bus"]],
            pgt=Float64(get(v,"pg",0.0))) for (k,v) in ref[:gen]],
      arc=[(i=k,rate_a=_rate(ref[:branch][l]),bus=busdict[i]) for (k,(l,i,j)) in enumerate(ref[:arcs])],
      branch=[begin f_idx=arcdict[i,br["f_bus"],br["t_bus"]]; t_idx=arcdict[i,br["t_bus"],br["f_bus"]];
           g,b=PowerModels.calc_branch_y(br); tr,ti=PowerModels.calc_branch_t(br); ttm=tr^2+ti^2;
           g_fr=br["g_fr"];b_fr=br["b_fr"];g_to=br["g_to"];b_to=br["b_to"];
           (i=branchdict[i],f_idx=f_idx,t_idx=t_idx,f_bus=busdict[br["f_bus"]],t_bus=busdict[br["t_bus"]],
            c1=(-g*tr-b*ti)/ttm,c2=(-b*tr+g*ti)/ttm,c3=(-g*tr+b*ti)/ttm,c4=(-b*tr-g*ti)/ttm,
            c5=(g+g_fr)/ttm,c6=(b+b_fr)/ttm,c7=(g+g_to),c8=(b+b_to),rate_a_sq=_rate_sq(br)) end for (i,br) in ref[:branch]],
      ref_buses=[busdict[i] for (i,k) in ref[:ref_buses]],
      vmax=[v["vmax"] for (k,v) in ref[:bus]],vmin=[v["vmin"] for (k,v) in ref[:bus]],
      pmax=[v["pmax"] for (k,v) in ref[:gen]],pmin=[v["pmin"] for (k,v) in ref[:gen]],
      qmax=[v["qmax"] for (k,v) in ref[:gen]],qmin=[v["qmin"] for (k,v) in ref[:gen]],
      rate_a=[_rate(ref[:branch][l]) for (k,(l,i,j)) in enumerate(ref[:arcs])],
      angmax=[b["angmax"] for (i,b) in ref[:branch]],angmin=[b["angmin"] for (i,b) in ref[:branch]])
    ca(x)=backend===nothing ? x : ExaModels.convert_array(x,backend)
    w=ExaModels.ExaCore(Float64; backend=backend)
    nb=length(data.bus); ng=length(data.gen)
    va=warm ? ExaModels.variable(w,nb;start=Float64[get(v,"va",0.0) for (k,v) in ref[:bus]]) : ExaModels.variable(w,nb)
    vm=ExaModels.variable(w,nb;start=(warm ? Float64[get(v,"vm",1.0) for (k,v) in ref[:bus]] : ones(nb)),lvar=data.vmin,uvar=data.vmax)
    pg=warm ? ExaModels.variable(w,ng;start=Float64[get(v,"pg",0.0) for (k,v) in ref[:gen]],lvar=data.pmin,uvar=data.pmax) : ExaModels.variable(w,ng;lvar=data.pmin,uvar=data.pmax)
    qg=warm ? ExaModels.variable(w,ng;start=Float64[get(v,"qg",0.0) for (k,v) in ref[:gen]],lvar=data.qmin,uvar=data.qmax) : ExaModels.variable(w,ng;lvar=data.qmin,uvar=data.qmax)
    _inf=fill(Inf,length(data.arc))
    p=ExaModels.variable(w,length(data.arc);lvar=(pq_box ? -data.rate_a : -_inf),uvar=(pq_box ? data.rate_a : _inf))
    q=ExaModels.variable(w,length(data.arc);lvar=(pq_box ? -data.rate_a : -_inf),uvar=(pq_box ? data.rate_a : _inf))
    if objective === :dist
        ExaModels.objective(w,dscale*(pg[g.i]-g.pgt)^2 for g in data.gen)
        ExaModels.objective(w,dscale*vw*(vm[b.i]-b.vmt)^2 for b in data.bus)
    elseif objective === :dist_ctrl
        ctrlb = [b for b in data.bus if b.isctrl]
        ExaModels.objective(w,dscale*(pg[g.i]-g.pgt)^2 for g in data.gen)
        ExaModels.objective(w,dscale*vw*(vm[b.i]-b.vmt)^2 for b in ctrlb)
    else
        ExaModels.objective(w,g.cost1*pg[g.i]^2+g.cost2*pg[g.i]+g.cost3 for g in data.gen)
    end
    ExaModels.constraint(w,va[i] for i in data.ref_buses)
    ExaModels.constraint(w,p[b.f_idx]-b.c5*vm[b.f_bus]^2-b.c3*(vm[b.f_bus]*vm[b.t_bus]*cos(va[b.f_bus]-va[b.t_bus]))-b.c4*(vm[b.f_bus]*vm[b.t_bus]*sin(va[b.f_bus]-va[b.t_bus])) for b in data.branch)
    ExaModels.constraint(w,q[b.f_idx]+b.c6*vm[b.f_bus]^2+b.c4*(vm[b.f_bus]*vm[b.t_bus]*cos(va[b.f_bus]-va[b.t_bus]))-b.c3*(vm[b.f_bus]*vm[b.t_bus]*sin(va[b.f_bus]-va[b.t_bus])) for b in data.branch)
    ExaModels.constraint(w,p[b.t_idx]-b.c7*vm[b.t_bus]^2-b.c1*(vm[b.t_bus]*vm[b.f_bus]*cos(va[b.t_bus]-va[b.f_bus]))-b.c2*(vm[b.t_bus]*vm[b.f_bus]*sin(va[b.t_bus]-va[b.f_bus])) for b in data.branch)
    ExaModels.constraint(w,q[b.t_idx]+b.c8*vm[b.t_bus]^2+b.c2*(vm[b.t_bus]*vm[b.f_bus]*cos(va[b.t_bus]-va[b.f_bus]))-b.c1*(vm[b.t_bus]*vm[b.f_bus]*sin(va[b.t_bus]-va[b.f_bus])) for b in data.branch)
    ExaModels.constraint(w,va[b.f_bus]-va[b.t_bus] for b in data.branch;lcon=data.angmin,ucon=data.angmax)
    ExaModels.constraint(w,p[b.f_idx]^2+q[b.f_idx]^2-b.rate_a_sq for b in data.branch;lcon=fill(-Inf,length(data.branch)))
    ExaModels.constraint(w,p[b.t_idx]^2+q[b.t_idx]^2-b.rate_a_sq for b in data.branch;lcon=fill(-Inf,length(data.branch)))
    c9=ExaModels.constraint(w,b.pd+b.gs*vm[b.i]^2 for b in data.bus)
    c10=ExaModels.constraint(w,b.qd-b.bs*vm[b.i]^2 for b in data.bus)
    ExaModels.constraint!(w,c9,a.bus=>p[a.i] for a in data.arc); ExaModels.constraint!(w,c10,a.bus=>q[a.i] for a in data.arc)
    ExaModels.constraint!(w,c9,g.bus=>-pg[g.i] for g in data.gen); ExaModels.constraint!(w,c10,g.bus=>-qg[g.i] for g in data.gen)
    nbus=length(ref[:bus]); ngen=length(ref[:gen]); model=ExaModels.ExaModel(w)
    return model,nbus,ngen,busdict,gendict,data
end
