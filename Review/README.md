# 嵌入式重难点总结

## 软中断流程

1. 通过`__swi(中断号) function`给中断函数指定中断号（一般在C代码中撰写）

2. 调用function时，相当于执行了`SWI 中断号`指令

3. 再执行`SWI`之前需要先注册中断程序，即把处理软中断的程序（SWI_handler）地址放到`0x08`软中断向量表地址上

4. 注册之后执行`SWI`的时候，会执行`0x08`中断向量地址上存放的跳转指令（`B SWI_handler `），即跳转到SWI_handler；同时将function的参数传递到寄存器内（如果参数不超过4个），超过4个应该存放到内存

   ```c
   // 中断注册程序
   unsigned Install_Handler (unsigned *handlerloc, unsigned *vector)
   {  
       unsigned vec, oldvec;
       vec = ((unsigned)handlerloc - (unsigned)vector - 0x8)>>2;	// 确保中断处理函数地址偏移在26位以内：正负32M
       if ((vec & 0xFF000000) != 0)	// 取低24位，即中断服务程序地址
        { return 0;}
       vec = 0xEa000000 | vec;			// 0xEa 应该时跳转指令B的操作码，这里表示跳转指令：”B handlerloc“
       oldvec = *vector;				 
       *vector = vec;					// 将中断向量表中断向量内写入跳转指令的编码，即vec
       return (oldvec);
   }
   ```

5. 跳转后，`SWI_handler`需要干五件事：

   - 1）将保存函数参数的寄存器以及lr寄存器存放到堆栈里保存，
   - 2）通过`BIC`指令获取中断号放置在r0寄存器，
   - 3）将堆栈指针保存到r1，
   - 4）跳转到要执行function的处理函数中（C_SWI_handle）加密，并将r0，r1寄存器作为函数参数传递
   - 5）处理完中断函数后恢复现场

```assembly
SWI_Handler
        STMFD      sp!,{r0-r12,lr}		;保存现场
        LDR        r0,[lr,#-4]          ;获取 SWI 指令
        BIC        r0,r0,#0xff000000    ;参数1，NUM
        MOV        R1, SP               ;参数2，传递堆栈指针
        BL C_SWI_Handler                ;To Function
        LDMFD      sp!, {r0-r12,pc}^	;处理完中断恢复现场，将最初的lr->pc，继续执行SWI指令的下一条指令
        END
```

6. `C_SWI_handle`接受中断号和堆栈指针，通过堆栈指针开始从堆栈中获取参数数据，根据中断号选择计算的程序逻辑，计算完成将结果再次写入堆栈

   ```c
   void C_SWI_handler (int swi_num, int *reg )
   {   switch (swi_num)
       {
           case 0 : 
               ……           /* SWI number 0 code */
                       break;
               case 1 :                 
               ……           /* SWI number 1 code */
                       break;
           ……
               default :   
               break；		/* Unknown SWI - report error */
       }
       return;
   }
   ```

   或

   ```assembly
   C_SWI_Handler 
           STMFD   sp!,{r0-r12,lr}
       	CMP    	r0,#MaxSWI          ; Range check
           LDRLE  	pc, [pc,r0,LSL #2] 	;(PC -> DCD SWInum0)
       	B       SWIOutOfRange 
   SWIJumpTable    
   		DCD    SWInum0
           DCD    SWInum1
   SWInum0   ; SWI number 0 code
           B    EndofSWI
   SWInum1   ; SWI number 1 code
           B    EndofSW
   EndofSW 
   		SUB 	lr, lr, #4
           LDMFD   sp!, {r0-r12,pc}^
           END
   ```

7. 最后从堆栈中取出计算结果放到寄存器r0中

8. 中断结束，从堆栈中恢复现场，继续执行`SWI`指令后的指令



