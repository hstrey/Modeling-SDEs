using IfElse

function squarewave(t,ton,toff, period)
    IfElse.ifelse(ton < mod(t,period) < toff,1.0 ,0.0)
end



