/*
 * Copyright 2016 - 2023  Angelo Matni, Simone Campanoni
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights to
 use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 of the Software, and to permit persons to whom the Software is furnished to do
 so, subject to the following conditions:

 * The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
 OR OTHER DEALINGS IN THE SOFTWARE.
 */
#include "arcana/noelle/core/LoopIterationSCC.hpp"
#include "arcana/noelle/core/ReductionSCC.hpp"
#include "arcana/noelle/core/SingleAccumulatorRecomputableSCC.hpp"
#include "arcana/gino/core/DOALL.hpp"
#include "arcana/gino/core/DOALLTask.hpp"
#include "arcana/noelle/core/Utils.hpp"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"
#include <vector>

namespace arcana::gino {

void DOALL::rewireLoopToIterateChunks(LoopContent *LDI, DOALLTask *task) {
  // N/T chunking notes:
  // We want to use N/T as a chunksize, where N is the number of loop iterations
  // known dynamically at task entry, and T is the number of cores obtained for
  // the task, known dynamically at task entry. In practice, we accept a
  // chunksize that is at minimum N/T and at maximum floor(N/T)+1. We derive
  // chunksize as N/T, and N as (end-start)/step + 1. Why the +1?
  //(end-start)/step expresses the number of steps between the start value and
  //(surpassing the) exit condition. However, there is always exactly one more
  // iteration across the execution of a loop than steps performed during the
  // execution. That is: iterations = total steps + 1. floor(
  // floor((end-start)/step) /T) + 1 is the exact chunksize we use.
  // floor((end-start)/step) is at least N-1. Then floor( (N-1) /T) is at least
  // (N/T)-1 and at most (N/T). Then, N/T <= floor( floor((end-start)/step) /T)
  // + 1 <= N/T + 1 N/T is always positive. Why? End is determined from the end
  // con. Start is determined from the start value. Thus end-start may be
  // positive or negative. However if start > end then stepsize is negative, and
  // positive if vice versa (we don't handle integer overflow code). Therefore,
  // N is always positive so N/T always positive.

  // Exit conditions.
  // The dynamic computation (end-start)/step produces N the dynamic number of
  // iterations. However... LGIV is start at the t=0'th entry to the header.
  // LGIV is start+step at the t=1'th entry to the header.
  // Consider end-start = N * step. That's another equation that can express N,
  // derived from (end-start)/step = N. Then end = start + N * step. start,
  // start + step, 6-0 = 6. step 2. 3 steps. But we exit during the iteration of
  // 6, there are 4 iterations run. Is it possible to screw things up with weird
  // core/chunk relations due to +1 terms instead of remainder operations? What
  // about preheader-header entry guards? Consider a loop that will have one
  // step, two iterations, and 2 cores are obtained for it.

  // RULE: this is doall so no loop-carried dependencies. Therefore, we win if
  // we have that all iters are executed by some core once and no extra iters
  // are executed. Change header entry guards to be "exit condition, but with
  // exit condition value + step" -- that'll allow the "exit condition iter" to
  // enter the loop if it's the first iter for some chunk.

  // In terms of exit logic, we should execute
  // code_executed_only_by_the_last_loop_iteration iff the exit condition is met
  // by the current iter.

  /*
   * Fetch loop and IV information.
   */
  auto invariantManager = LDI->getInvariantManager();
  auto loopSummary = LDI->getLoopStructure();
  auto loopHeader = loopSummary->getHeader();
  auto loopPreHeader = loopSummary->getPreHeader();
  auto preheaderClone = task->getCloneOfOriginalBasicBlock(loopPreHeader);
  auto headerClone = task->getCloneOfOriginalBasicBlock(loopHeader);
  auto allIVInfo = LDI->getInductionVariableManager();

  /*
   * Collect clones of step size deriving values for all induction variables
   * of the parallelized loop. DD: that is to say, the value which represents
   * the final derivation of the step size. So the value can be treated as the
   * step size.
   */
  IRBuilder<> chunkSizeBuilder(task->getEntry());
  IRBuilder<> entryBuilder(task->getEntry());
  Instruction *firstChunkSizeInst = nullptr;
  auto jumpEntry =
      task->getEntry()->getTerminator(); // br from task-imposed entry block to
                                         // actual loop header
  chunkSizeBuilder.SetInsertPoint(jumpEntry);
  auto clonedStepSizeMap =
      this->cloneIVStepValueComputation(LDI, 0, chunkSizeBuilder);

  /*
   * Get the end and start values for the LGIV to compute N/T
   */
  auto lgivForChunkingVal = allIVInfo->getLoopGoverningInductionVariable();
  auto typeOfLGIV = lgivForChunkingVal->getInductionVariable()->getType();
  auto endConVal = lgivForChunkingVal->getExitConditionValue();
  auto endConValClone = this->fetchCloneInTask(task, endConVal);
  auto lgivStartVal = lgivForChunkingVal->getInductionVariable()
                          ->getLoopEntryPHI()
                          ->getIncomingValueForBlock(
                              loopPreHeader); // lgiv start val is the lgiv val
                                              // in the preheader, nothing more
  auto lgivStartClone = this->fetchCloneInTask(task, lgivStartVal);
  auto stepOfLGIVForChunkSize =
      clonedStepSizeMap.at(lgivForChunkingVal->getInductionVariable());
  assert(lgivStartClone && "No lgiv start value??");
  assert(endConValClone && "No end con val??");
  auto endMinusStart = chunkSizeBuilder.CreateSub(endConValClone,
                                                  lgivStartClone,
                                                  "endMinusStart");
  std::vector<Value *> printVals1;
  // printVals1.push_back(endConValClone);
  // printVals1.push_back(lgivStartClone);
  // printVals1.push_back(endMinusStart);
  // printVals1.push_back(task->taskInstanceID);
  // arcana::noelle::Utils::injectPrint(lgivStartClone, "%d\n",
  // chunkSizeBuilder); arcana::noelle::Utils::injectPrint(printVals1, "end: %d,
  // start: %d, endMinusStart: %d, TID: %lld\n", chunkSizeBuilder);

  /*
   * Compute chunksize as floor( floor((end-start)/step) /T) + 1, and assign
   * that number to usedChunkSize. Floor operations are expressed implicitly in
   * SDiv instrs.
   */
  firstChunkSizeInst = dyn_cast<Instruction>(
      endMinusStart); // tracks where chunkSize computation begins, so we can
                      // set IRBuilder insert location later
  auto S = chunkSizeBuilder.CreateSDiv(endMinusStart,
                                       stepOfLGIVForChunkSize,
                                       "N"); //= floor((end-start) / step)
  if (!firstChunkSizeInst) {
    firstChunkSizeInst = dyn_cast<Instruction>(S);
  }
  auto N = chunkSizeBuilder.CreateAdd(S, ConstantInt::get(S->getType(), 1));
  if (!firstChunkSizeInst) {
    firstChunkSizeInst = dyn_cast<Instruction>(N);
  }
  auto T = task->getTaskBody()->getArg(
      2);       // this should be correct. See DOALL_linker.cpp
  Value *usedT; // check if T requires trunc/ext
  if (T->getType() != typeOfLGIV) {
    usedT = chunkSizeBuilder.CreateTrunc(T, N->getType());
    if (!firstChunkSizeInst) {
      firstChunkSizeInst = dyn_cast<Instruction>(usedT);
    }
  } else {
    usedT = T;
  }
  auto NT = chunkSizeBuilder.CreateSDiv(
      N,
      usedT,
      "NoverT"); //= floor( floor((end-start) / step) /T)
  if (!firstChunkSizeInst) {
    firstChunkSizeInst = dyn_cast<Instruction>(NT);
  }
  auto chunkSizeDD = chunkSizeBuilder.CreateAdd(
      NT,
      ConstantInt::get(NT->getType(), 1),
      "remainderFix"); //= floor( floor((end-start)/step) /T) + 1
  if (!firstChunkSizeInst) {
    firstChunkSizeInst = dyn_cast<Instruction>(chunkSizeDD);
  }
  Value *usedChunkSize; // Value used for all chunksize computations later
  if (chunkSizeDD->getType() != task->taskInstanceID->getType()) {
    usedChunkSize = chunkSizeBuilder.CreateZExt(chunkSizeDD,
                                                task->taskInstanceID->getType(),
                                                "usedChunkSize");
  } else {
    usedChunkSize = chunkSizeDD;
  }

  std::vector<Value *> printVals;
  // printVals.push_back(chunkSizeDD);
  // printVals.push_back(N);
  // printVals.push_back(usedT);
  // printVals.push_back(stepOfLGIVForChunkSize);
  // printVals.push_back(NT);
  // printVals.push_back(task->taskInstanceID);

  // arcana::noelle::Utils::injectPrint(printVals, "ChunkSizeDD: %d, N: %d,
  // usedT: %d, stepOfLGIV: %d, NT: %d, TID: %lld\n", chunkSizeBuilder);
  auto chunkCounterType = usedChunkSize->getType();
  Value *stepSizeForEndValCalc; // Reconcile types between step and chunksize
  if (usedChunkSize->getType() != stepOfLGIVForChunkSize->getType()) {
    stepSizeForEndValCalc =
        chunkSizeBuilder.CreateSExt(stepOfLGIVForChunkSize,
                                    usedChunkSize->getType());
  } else {
    stepSizeForEndValCalc = stepOfLGIVForChunkSize;
  }
  auto chunkSizeXStepSize = chunkSizeBuilder.CreateMul(usedChunkSize,
                                                       stepSizeForEndValCalc,
                                                       "chunkSizeTimesStep");

  // Below will need cleanup -- keeping code as comment is currently valuable
  // for seeing what's been changed

  // auto print = task->getTaskBody()->getParent()->getOrInsertFunction()

  /*
   * Generate PHI to track progress on the current chunk
   */
  auto jumpToLoop =
      task->getEntry()->getTerminator(); // br from task-imposed entry block to
                                         // actual loop preheader
  // Task 2
  /*auto chunkPHI = IVUtility::createChunkPHI(
      preheaderClone, // so what this does is it dumps basically a "n = n + 1"
                      // in the latch, and a "if n == CHUNKSIZE then set n = 0"
      headerClone, // via a select in the latch + a phi in the header. So the
                   // phi is 0 if CHUNKSIZE iterations have been done, or
      chunkCounterType, // if it was reached from the preheader.
      usedChunkSize);
  */

  entryBuilder.SetInsertPoint(jumpEntry);
  // entryBuilder.SetInsertPoint((Instruction*) endMinusStart);

  /*
   * Determine start value of the IV for the task
   * The start value of an IV depends on the first iteration executed by a task.
   * This value, for a given task, is
   *    = original_start + (original_step_size * chunk_size * task_instance_id)
   *
   * where task_instance_id is the dynamic ID that spawn tasks will have, which
   * start at 0 (for the first task instance), 1 (for the second task instance),
   * until N-1 (for the last task instance).
   */
  // task 2: we keep this
  Value *lgivEndVal; // Task 2
  for (auto ivInfo : allIVInfo->getInductionVariables(*loopSummary)) {
    auto startOfIV = this->fetchCloneInTask(
        task,
        ivInfo->getStartValue()); // DD: I assume this literally gets the LLVM
                                  // value that the IV starts at
    auto stepOfIV =
        clonedStepSizeMap.at(ivInfo); // per earlier this is the cloned LLVM
                                      // value that is the step size
    auto loopEntryPHI =
        ivInfo->getLoopEntryPHI(); // this must exist else the IV would be a
                                   // value local to the loop. Since it must be
                                   // initialized somewhere, that can't be true
                                   // -- at most there can be a phi of (0, val)
                                   // at the header, e.g.
    auto ivPHI = cast<PHINode>(this->fetchCloneInTask(
        task,
        loopEntryPHI)); // get the clone of the loopEntryPhi

    auto nthCoreOffset =
        IVUtility::scaleInductionVariableStep( // returns a multiply of step and
                                               // arg #4 -- step * coreidx *
                                               // chunksize
            preheaderClone,
            ivPHI,
            stepOfIV,
            entryBuilder.CreateMul(task->taskInstanceID,
                                   usedChunkSize,
                                   "coreIdx_X_chunkSize"));

    auto offsetStartValue = IVUtility::offsetIVPHI(
        preheaderClone,
        ivPHI,
        startOfIV,
        nthCoreOffset); // creates an add of startOfIV and nthCoreOffset,
                        // matching type of ivPHI, without modifying ivPHI at
                        // all.

    offsetStartValue->setName("offsetStartValue");
    ivPHI->setIncomingValueForBlock( // set the ivPHI to use offsetStartValue
        preheaderClone,
        offsetStartValue);

    // Task 2: we should calculate an lgiv end value based on each chunk.
    if (ivInfo
        == allIVInfo->getLoopGoverningInductionVariable()
               ->getInductionVariable()) {
      if (offsetStartValue->getType() != usedChunkSize->getType()) {
        auto truncOffChunkSize =
            entryBuilder.CreateTrunc(chunkSizeXStepSize,
                                     offsetStartValue->getType());
        auto lgivEndValTemp = entryBuilder.CreateAdd(
            truncOffChunkSize, // go ahead 1 chunksize of steps
            offsetStartValue,
            "lgivEndValTemp");
        auto lgivEndValFin =
            entryBuilder.CreateSub(lgivEndValTemp, // step back 1 step
                                   clonedStepSizeMap[ivInfo],
                                   "lgivEndValFin");
        lgivEndVal = lgivEndValFin;
      } else {
        auto lgivEndValTemp = entryBuilder.CreateAdd(chunkSizeXStepSize,
                                                     offsetStartValue,
                                                     "lgivEndValTemp");
        auto lgivEndValFin = entryBuilder.CreateSub(lgivEndValTemp,
                                                    clonedStepSizeMap[ivInfo],
                                                    "lgivEndValFin");
        lgivEndVal = lgivEndValFin;
      }
      std::vector<Value *> printVals2;
      printVals2.push_back(offsetStartValue);
      printVals2.push_back(lgivEndVal);
      printVals2.push_back(task->taskInstanceID);

      // arcana::noelle::Utils::injectPrint(printVals2, "offsetStartValue: %d,
      // lgivEndVal: %d, TID: %lld\n", entryBuilder);
    }
  }

  /*
   * Remove the old terminator because it will replace with the check.
   */

  for (auto latch : loopSummary->getLatches()) {
    auto cloneLatch = task->getCloneOfOriginalBasicBlock(latch);
    auto latchTerminator = cloneLatch->getTerminator();
    latchTerminator->eraseFromParent();
    IRBuilder<> latchBuilder(cloneLatch);
    auto lgivPHI = allIVInfo->getLoopGoverningInductionVariable(*loopSummary)
                       ->getInductionVariable()
                       ->getLoopEntryPHI();
    auto lgivPHIClone = cast<PHINode>(this->fetchCloneInTask(task, lgivPHI));
    // Task 2: to handle exit logics correctly, we need to record which lgiv
    // value to use for each exiting block. For each block where we insert a
    // chunkDoneCheck the correct value if the block's successor edge is the
    // exit edge is the producer of the lgiv value for the next iteration.

    for (auto i = 0u; i < lgivPHIClone->getNumIncomingValues(); ++i) {
      auto B = lgivPHIClone->getIncomingBlock(i);
      if (preheaderClone == B)
        continue;

      auto initialLatchValue = lgivPHIClone->getIncomingValue(i);
      this->IVValueJustBeforeEnteringBody[lgivPHIClone].insert(
          initialLatchValue);
    }

    auto chunkDoneCheck = latchBuilder.CreateCmp(CmpInst::ICMP_EQ,
                                                 lgivPHIClone,
                                                 lgivEndVal,
                                                 "chunkDoneCheck");
    latchBuilder.CreateCondBr(chunkDoneCheck,
                              task->getLastBlock(0),
                              headerClone);
  }

  /*
   * Determine additional step size
   *   from the beginning of the chunk that will be executed by the next task
   *   to the start of the next chunk that task-instance will execute.
   * The step size is this:
   *   chunk_step_size: original_step_size * (num_task_instances - 1) *
   * chunk_size
   */
  for (auto ivInfo : allIVInfo->getInductionVariables(*loopSummary)) {
    auto stepOfIV = clonedStepSizeMap.at(ivInfo);
    auto cloneLoopEntryPHI = this->fetchCloneInTask(
        task,
        ivInfo->getLoopEntryPHI()); // fetch the cloned loop entry phi, per IV.
                                    // So there's one cloned LEPHI per IV. This
                                    // makes sense because ince it must be
                                    // initialized somewhere, that can't be true
                                    // -- at most there can be a phi of (0, val)
                                    // at the header, e.g.
    assert(cloneLoopEntryPHI != nullptr);
    auto ivPHI = cast<PHINode>(cloneLoopEntryPHI);
    auto onesValueForChunking = ConstantInt::get(chunkCounterType, 1);
    auto chunkStepSize = IVUtility::scaleInductionVariableStep(
        preheaderClone,
        ivPHI,
        stepOfIV,
        entryBuilder.CreateMul(
            entryBuilder.CreateSub(task->numTaskInstances,
                                   onesValueForChunking,
                                   "numCoresMinus1"),
            usedChunkSize,
            "numCoresMinus1_X_chunkSize")); // build a multiply between
                                            // (N-1)(sizeof(chunk)) where N
                                            // number of cores.

    // so what this does is, it constructs a selectinst where the condition is
    // whether the chunk is finished, if chunk is finished the selectinst adds
    // chunksize to the IV's value and outputs the result, if chunk is NOT
    // finished the selectinst outputs the IV's existing value. The select is
    // placed ahead of the terminator of the BB containing ivPHI. So, that's
    // whatever BB loopentryphi is in...
    // task 2:
    /*
    auto chunkedIVValues = IVUtility::chunkInductionVariablePHI(preheaderClone,
                                                                ivPHI,
                                                                chunkPHI,
                                                                chunkStepSize);
    this->IVValueJustBeforeEnteringBody[ivPHI] = chunkedIVValues;
    */
  }
  entryBuilder.SetInsertPoint(jumpEntry);
  /*
   * Fetch the SCCDAG of the loop.
   */
  auto sccManager = LDI->getSCCManager();
  auto sccdag = sccManager->getSCCDAG();

  /*
   * Generates code for periodic variable SCCs to match the DOALL chunking
   * strategy.
   */
  for (auto scc : sccdag->getSCCs()) {
    auto sccInfo = sccManager->getSCCAttrs(scc);
    auto periodicVariableSCC = dyn_cast<PeriodicVariableSCC>(sccInfo);
    if (periodicVariableSCC == nullptr)
      continue;

    if (this->verbose >= Verbosity::Maximal) {
      errs()
          << "DOALL: periodic variable with initial value "
          << *periodicVariableSCC->getInitialValue() << "\n"
          << "       and period " << *periodicVariableSCC->getPeriod() << "\n"
          << "       and step " << *periodicVariableSCC->getStepValue() << "\n";
    }

    /*
     * Retrieve the relevant Values for the periodic variable SCC.
     */
    auto initialValue = periodicVariableSCC->getInitialValue();
    auto period = periodicVariableSCC
                      ->getPeriod(); // period is necessarily an actual value,
                                     // otherwise it can't be computed with
    auto step =
        periodicVariableSCC
            ->getStepValue(); // period is the modulo vs step is the delta
    auto phi =
        periodicVariableSCC
            ->getPhiThatAccumulatesValuesBetweenLoopIterations(); // SingleAccumulatorRecomputableSCC
                                                                  // have a
                                                                  // single
                                                                  // PHInode.
                                                                  // periodicVariableSCC
                                                                  // ->
                                                                  // SingleAccumulatorRecomputableSCC,
                                                                  // and this is
                                                                  // the
                                                                  // phinode.
    assert(
        phi->getNumIncomingValues() == 2
        && "DOALL: PHINode in periodic variable SCC doesn't have exactly two entries!"); // so this is here, I'm guessing, because we don't like the notion of a periodicVariableSCC inside a loop that has more than one latch.
    auto taskPHI = cast<PHINode>(task->getCloneOfOriginalInstruction(
        phi)); // get the task's clone of the previous

    // check which index of the phi is the entry block and which is the loop
    // block.
    uint64_t entryBlock = 0;
    uint64_t loopBlock = 0;
    if (phi->getIncomingValue(0) == initialValue) {
      entryBlock = 0;
      loopBlock = 1;
    } else {
      assert(phi->getIncomingValue(1) == initialValue
             && "DOALL: periodic variable SCC selected the wrong PHINode!");
      entryBlock = 1;
      loopBlock = 0;
    }
    auto taskLoopBlock = task->getCloneOfOriginalBasicBlock(
        phi->getIncomingBlock(loopBlock)); // latch.
    assert(taskLoopBlock != nullptr);
    auto loopValue =
        phi->getIncomingValue(loopBlock); // the value assigned to phi if phi is
                                          // reached from inside loop
    assert(isa<Instruction>(loopValue));
    auto taskLoopValue = task->getCloneOfOriginalInstruction(
        cast<Instruction>(loopValue)); // clone of previous
    assert(taskLoopValue != nullptr);

    /*
     * Calculate the periodic variable's initial value for the task.
     * This value is: initialValue + step_size * ((task_id * chunk_size) %
     * period)
     */
    auto coreIDxChunkSize = entryBuilder.CreateMul(task->taskInstanceID,
                                                   usedChunkSize,
                                                   "coreIdx_X_chunkSize");
    auto numSteps = entryBuilder.CreateSRem(
        coreIDxChunkSize,
        period,
        "numSteps"); // size of chunk module period produces step difference
    auto numStepsTrunc = entryBuilder.CreateTrunc(
        numSteps,
        step->getType()); // presumably handles oversize periods or something
    auto numStepsxStepSize = entryBuilder.CreateMul(
        step,
        numStepsTrunc,
        "stepSize_X_numSteps"); // per name, but num steps is again chunk module
                                // period
    auto numStepsxStepSizeTrunc =
        entryBuilder.CreateTrunc(numStepsxStepSize, initialValue->getType());
    auto chunkInitialValue = entryBuilder.CreateAdd(initialValue,
                                                    numStepsxStepSizeTrunc,
                                                    "initialValuePlusStep");
    taskPHI->setIncomingValue(
        entryBlock,
        chunkInitialValue); // alter the phi at the top of the loop so that the
                            // initial val is the thing we compute here

  } // task 2

  /*
  /
   * Determine value of the start of this core's next chunk
   * from the beginning of the next core's chunk.
   * Formula: (next_chunk_initialValue + (step_size * (num_cores - 1) *
   * chunk_size)) % period
   /
  auto onesValueForChunking = ConstantInt::get(chunkCounterType, 1);
  auto numCoresMinus1 = entryBuilder.CreateSub(task->numTaskInstances,
                                               onesValueForChunking,
                                               "numCoresMinus1");
  auto chunkStepSize = entryBuilder.CreateMul(numCoresMinus1,
                                              usedChunkSize,
                                              "numCoresMinus1_X_chunkSize");
  auto chunkStepSizeTrunc = entryBuilder.CreateTrunc(
      chunkStepSize,
      step->getType()); // why? Might be just a typing thing but can you fuck
                        // up the modulo with this?
  auto chunkStep = entryBuilder.CreateMul(
      chunkStepSizeTrunc,
      step,
      "chunkStep"); // Note that we shoved all this computation in the entry
                    // block, so it only runs once

  /
   * Add the instructions for the calculation of the next chunk's start value
   * in the loop's body.
   /
  IRBuilder<> loopBuilder(
      taskLoopBlock); // recall that taskLoopBlock is the latch
  loopBuilder.SetInsertPoint(
      taskLoopBlock
          ->getTerminator()); // So we're at the bottom of the iteration which
                              // is why we're in the latch, and we have to be
                              // there because we've gotta be before the
                              // superstep into the next set of iterations.
  auto chunkStepTrunc =
      loopBuilder.CreateTrunc(chunkStep, taskLoopValue->getType());
  auto nextChunkValueBeforeMod =
      loopBuilder.CreateAdd(taskLoopValue,
                            chunkStepTrunc,
                            "nextChunkValueBeforeMod");
  auto periodTrunc =
      loopBuilder.CreateTrunc(period, taskLoopValue->getType());
  auto nextChunkValue = loopBuilder.CreateSRem(
      nextChunkValueBeforeMod,
      periodTrunc,
      "nextChunkValue"); // srem: signed division. Modulo stand-in. Should it
                         // be UREM @simone?

  /
   * Determine if we have reached the end of the chunk, and choose the
   * periodic variable's next value accordingly.
   /
  auto isChunkCompleted =
      cast<SelectInst>(chunkPHI->getIncomingValueForBlock(taskLoopBlock))
          ->getCondition(); // sure, literally build a select inst where
                            // condition is "isChunkCompleted" and return is
                            // nextChunkValue or the extant taskLoopValue
  auto nextValue = loopBuilder.CreateSelect(
      isChunkCompleted, // still inserting into the latch
      nextChunkValue,
      taskLoopValue,
      "nextValue");
  taskPHI->setIncomingValueForBlock(taskLoopBlock, nextValue);
}

/*
 * The exit condition needs to be made non-strict to catch iterating past it
 */
  auto loopGoverningIVAttr = allIVInfo->getLoopGoverningInductionVariable();
  LoopGoverningIVUtility ivUtility(loopSummary,
                                   *allIVInfo,
                                   *loopGoverningIVAttr);
  auto cmpInst = cast<CmpInst>(task->getCloneOfOriginalInstruction(
      loopGoverningIVAttr
          ->getHeaderCompareInstructionToComputeExitCondition()));
  auto brInst = cast<BranchInst>(task->getCloneOfOriginalInstruction(
      loopGoverningIVAttr->getHeaderBrInst()));
  auto basicBlockToJumpToWhenTheLoopEnds = task->getLastBlock(0);
  ivUtility.updateConditionAndBranchToCatchIteratingPastExitValue(
      cmpInst,
      brInst,
      basicBlockToJumpToWhenTheLoopEnds);
  auto updatedCmpInst = cmpInst;

  /*
   * The exit condition value does not need to be computed each iteration and so
   * the value's derivation can be hoisted into the preheader.
   *
   * Instructions that the PDG states are independent can include PHI nodes.
   * Assert that any PHIs are invariant. Hoist one of those values (if
   * instructions) to the preheader.
   */
  // DD: task 2: this can stay but I assume this should be like, a thing, kind
  // of... independent of chunking strategy basically.
  auto exitConditionValue =
      this->fetchCloneInTask(task,
                             loopGoverningIVAttr->getExitConditionValue());
  assert(exitConditionValue != nullptr);

  chunkSizeBuilder.SetInsertPoint(firstChunkSizeInst);
  if (auto exitConditionInst = dyn_cast<Instruction>(exitConditionValue)) {
    auto &derivation = ivUtility.getConditionValueDerivation();
    for (auto I : derivation) {
      assert(
          invariantManager->isLoopInvariant(I)
          && "DOALL exit condition value is not derived from loop invariant values!");

      /*
       * Fetch the clone of @I
       */
      auto cloneI = task->getCloneOfOriginalInstruction(I);

      if (auto clonePHI = dyn_cast<PHINode>(cloneI)) {
        auto usedValue = clonePHI->getIncomingValue(0);
        clonePHI->replaceAllUsesWith(
            usedValue); // DD: the legality of these ops is guaranteed by
                        // invariantManager->isLoopInvariant
        clonePHI
            ->eraseFromParent(); // DD: come to think of it, isLoopInvariant
                                 // proly relies on "phi has only 1 val" or smth
        cloneI = dyn_cast<Instruction>(usedValue);
        if (!cloneI) {
          continue;
        }
      }

      cloneI->removeFromParent();
      chunkSizeBuilder.Insert(cloneI); // move to entry
    }

    exitConditionInst->removeFromParent();
    chunkSizeBuilder.Insert(exitConditionInst);
    // DD: overall this really just does what the parent comment says
  }

  /*
   * NOTE: When loop governing IV attribution allows for any other instructions
   * in the header other than those of the IV and its comparison, those
   * unrelated instructions should be copied into the body and the exit block
   * (to preserve the number of times they execute)
   *
   * The logic in the exit block must be guarded so only the "last" iteration
   * executes it, not any cores that pass the last iteration. This is further
   * complicated because the mapping of live-out environment producing
   * instructions might need to be updated with the peeled instructions in the
   * exit block
   *
   * A temporary mitigation is to transform loop latches with conditional
   * branches that verify if the next iteration would ever occur. This still
   * requires live outs to be propagated from both the header and the latches
   */

  // DD: why is this a temporary mitigation??? Does this even exist? What?

  /*
   * Identify any instructions in the header that are NOT sensitive to the
   * number of times they execute: 1) IV instructions, including the comparison
   * and branch of the loop governing IV 2) The PHI used to chunk iterations 3)
   * Any PHIs of reducible variables 4) Any loop invariant instructions that
   * belong to independent-execution SCCs
   */
  std::unordered_set<Instruction *> repeatableInstructions;
  std::unordered_set<Instruction *> reducibleHeaderPHIsWithHeaderLogic;

  /*
   * Collect IV instructions, including the comparison
   * and branch of the loop governing IV by iterating the
   * InductionVariableManager
   */
  for (auto ivInfo : allIVInfo->getInductionVariables(*loopSummary)) {
    for (auto I : ivInfo->getAllInstructions()) {
      repeatableInstructions.insert(task->getCloneOfOriginalInstruction(I));
    }
  }
  repeatableInstructions.insert(cmpInst);
  repeatableInstructions.insert(brInst);

  /*
   * Collect The PHI used to chunk iterations
   */
  // Task 2: doesn't exist
  // repeatableInstructions.insert(chunkPHI);

  /*
   * Collect Any PHIs of reducible variables by identifying all reducible SCCs
   */
  auto nonDOALLSCCs = sccManager->getSCCsWithLoopCarriedDataDependencies();
  for (auto sccInfo : nonDOALLSCCs) {
    auto reductionSCC = dyn_cast<ReductionSCC>(sccInfo);
    if (reductionSCC == nullptr)
      continue;

    auto headerPHI =
        reductionSCC->getPhiThatAccumulatesValuesBetweenLoopIterations();
    assert(headerPHI != nullptr);

    auto hasInstsInHeader = false;
    auto scc = sccInfo->getSCC();
    for (auto nodePair : scc->internalNodePairs()) {
      auto value = nodePair.first;
      auto inst = cast<Instruction>(value);
      if (inst->getParent() != loopHeader)
        continue;

      auto instClone = task->getCloneOfOriginalInstruction(inst);
      repeatableInstructions.insert(instClone);
      hasInstsInHeader = true;
    }

    if (hasInstsInHeader) {
      auto headerPHIClone = task->getCloneOfOriginalInstruction(headerPHI);
      reducibleHeaderPHIsWithHeaderLogic.insert(headerPHIClone);
    }
  }

  /*
   * Collect any loop invariant instructions that belong to
   * independent-execution SCCs by identifying header instructions belonging to
   * independent SCCs that are loop invariant
   */
  for (auto &I : *loopHeader) {
    auto scc = sccdag->sccOfValue(&I);
    auto sccInfo = sccManager->getSCCAttrs(scc);
    if (!isa<LoopIterationSCC>(sccInfo))
      continue;

    auto isInvariant = invariantManager->isLoopInvariant(&I);
    if (!isInvariant)
      continue;

    repeatableInstructions.insert(task->getCloneOfOriginalInstruction(&I));
  }

  /*
   * Fetch the required information to generate any extra condition code needed.
   */
  auto loopGoverningIV = loopGoverningIVAttr->getInductionVariable();
  auto loopGoverningPHI =
      task->getCloneOfOriginalInstruction(loopGoverningIV->getLoopEntryPHI());
  auto origValueUsedToCompareAgainstExitConditionValue =
      loopGoverningIVAttr->getValueToCompareAgainstExitConditionValue();
  auto valueUsedToCompareAgainstExitConditionValue =
      task->getCloneOfOriginalInstruction(
          origValueUsedToCompareAgainstExitConditionValue);
  assert(valueUsedToCompareAgainstExitConditionValue != nullptr);
  auto stepSize = clonedStepSizeMap.at(loopGoverningIV);

  /*
   * Check if we need to check whether we need to add a condition to execute
   * instructions in the new header for tasks that are executing the header in
   * iterations after the last one.
   */
  auto requiresConditionBeforeEnteringHeader = false;
  for (auto &I : *headerClone) {
    if (repeatableInstructions.find(&I) == repeatableInstructions.end()) {
      requiresConditionBeforeEnteringHeader = true;
      break;
    }
  }
  if (!requiresConditionBeforeEnteringHeader) {
    // Task 2 but also in general. Like... I'm convinced not doing this
    // creates a race condition when we're in the only-repeatables world
    // and don't have header guards. We seem to be able to spawn task threads
    // that enter the loop header with an LGIV value beyond the end condition,
    // and those instantly go to the block containing isLastLoopIteration,
    // and they PASS THE COMPARE! THEY PASS THE COMPARE! THEY STORE AFTERWARDS!

    // return; // Task 2: don't put preheader guards.
  }
  /*  if (!requiresConditionBeforeEnteringHeader) {

      *
       * We have to handle the special case where there are reducible SCCs
    (i.e.,
       * reducable variables at the source code level) for which some of the
       * non-PHI instructions are also contained in the header. For example,
       * consider the following code:
       *
       * BB0:
       *   ...
       *   br %BB1
       *
       * BB1:
       *  %v2 = PHI [%v1, BB1], [%v0, BB0]
       *  ...
       *  %v1 = add %v2, 1
       *  ...
       *  br %c %BB1, %BB2
       *
       * BB2:
       *  return
       *
       *
       * This is a special case because there are two values that we could use
    to
       * store into the reduction variable:
       * 1) the PHI instruction (e.g., %v2)
       * 2) the non-PHI instruction that does the accumulation (e.g., %v1) We
    need
       * to use the right value depending on whether the header would NOT have
       * executed its last iteration. If that is the case, then we need to use
    the
       * PHI instruction. Otherwise, if the last instance of the header was
    meant
       * to be executed, then we need to use the non-PHI instruction.fd
       *
       * To solve this problem, we are going to inject a new SelectInst that
       * checks whether the last execution of the header was meant to be
    executed.
       * This SelectInst will be inserted into the basic block that leaves the
       * task, just before storing the right value into the reduction variable
    of
       * the current task.
       *
       //DD: really what this does is, it takes a special case of Nonrepeatable
    World
       //where we're allowed to hoist the checking of (iteration meant to be
    executed)
       //to after the loop completes.
      auto env = LDI->getEnvironment();
      auto envUser = this->envBuilder->getUser(0);
      std::vector<std::pair<Instruction *, Instruction *>>
          headerPHICloneAndProducerPairs;
      for (auto envID : envUser->getEnvIDsOfLiveOutVars()) {

        *
         * Fetch the clone of the producer of the current live-out variable.
         * Fetch the header PHI of the live-out variable.
         *
        auto producer = cast<Instruction>(env->getProducer(envID));
        assert(producer != nullptr);
        auto scc = sccdag->sccOfValue(producer);
        assert(scc != nullptr);
        auto sccInfo = sccManager->getSCCAttrs(scc);
        assert(sccInfo != nullptr);

        *
         * Check if the current live-out variable is reducible
         *
         * Live-out variables that are not reducible are handled separetely by
         * "generateCodeToStoreLiveOutVariables"
         *
        if (!isa<ReductionSCC>(sccInfo)) {
          continue;
        }
        auto reductionSCC = cast<ReductionSCC>(sccInfo);

        *
         * Check whether the header PHI is part of the set of PHIs we need to
         * guard
         *
        auto headerPHI =
            reductionSCC->getPhiThatAccumulatesValuesBetweenLoopIterations();
        assert(headerPHI != nullptr);
        auto clonePHI = task->getCloneOfOriginalInstruction(headerPHI);
        assert(clonePHI != nullptr);
        if (reducibleHeaderPHIsWithHeaderLogic.find(clonePHI)
            != reducibleHeaderPHIsWithHeaderLogic.end()) {
          headerPHICloneAndProducerPairs.push_back(
              std::make_pair(clonePHI, producer));
        }
      }

      *
       * Produce exit block SelectInst for all reducible SCCs that have header
       * logic
       *
      if (headerPHICloneAndProducerPairs.size() > 0) {
        auto startValue =
            this->fetchCloneInTask(task, loopGoverningIV->getStartValue());

        *
         * If this comment gets its spacing normalized, it won't read right.
         * Reminder: must use non-phi iff header was meant to execute its last
         * iteration.
         *
         * Piece together the condition for all the SelectInst:
         * ((prev loop-governing IV's value triggered exiting the loop) && (IV
         * //DD: using "prev LGIV value" at any point is sacriledge. In this
    case
         * it seems to mean "n-1" but I intially took it to mean "n-chunkstep-1"
         * header PHI != start value)) ? header phi // this will contain the
         * pre-header value or the previous latch value : original producer
    //this
         * will be the live out value from the header
         *
        IRBuilder<> exitBuilder(
            task->getLastBlock(0)->getFirstNonPHIOrDbgOrLifetime());
        auto prevIterationValue =
            ivUtility.generateCodeToComputeValueToUseForAnIterationAgo(
                exitBuilder,
                loopGoverningPHI,
                stepSize);
        auto headerToExitCmp = updatedCmpInst->clone();
        headerToExitCmp->replaceUsesOfWith(
            valueUsedToCompareAgainstExitConditionValue,
            prevIterationValue);
        exitBuilder.Insert(headerToExitCmp);
        auto wasNotFirstIteration =
            exitBuilder.CreateICmpNE(loopGoverningPHI, startValue);
        auto skipLastHeader =
            exitBuilder.CreateAnd(wasNotFirstIteration, headerToExitCmp);

        *
         * Use SelectInst created above to propagate the correct live out value
         * for all reducible SCCs that have header logic
         *
        for (auto pair : headerPHICloneAndProducerPairs) {
          auto headerPHIClone = pair.first;
          auto producer = pair.second;
          auto producerClone = task->getCloneOfOriginalInstruction(producer);
          auto lastReducedInst =
              cast<Instruction>(exitBuilder.CreateSelect(skipLastHeader,
                                                         headerPHIClone,
                                                         producerClone));

          /
           * HACK: Replace original producer clone entry with new SelectInst
           * What would be cleaner is to invoke task->addLiveOut(producer,
           * lastReducedInst) but this would require ParallelizationTechnique to
           * support the possibility that its internal liveOutClones map could
           * contain values with no equivalent in the original live out SCC.
           * TODO: Update
           * fetchOrCreatePHIForIntermediateProducerValueOfReducibleLiveOutVariable
           * to support finding potentially newly created values that are
    inserted
           * into the liveOutClones map via the addLiveOut API
           /
          task->addInstruction(producer, lastReducedInst);
        }
      }

      /
       * There is no need for pre-header / latch guards, so we return
       * TODO: Isolate reducible live out guards and pre-header / latch guards
    to
       * helper methods so this function's control flow is simpler
       /
      return;
    }

    /
     * The new header includes instructions that should be executed only if we
     * know that we didn't pass the last iteration. Hence, we need to add code
    to
     * check this condition before entering the header. Such code needs to be
     * added for all predecessors of the header: pre-header and latches.
     /

    /
     * In each latch, check whether we passed the last iteration.
     /
    for (auto latch : loopSummary->getLatches()) {

      /
       * Fetch the latch in the loop within the task.
       /
      auto cloneLatch = task->getCloneOfOriginalBasicBlock(latch);

      /
       * Remove the old terminator because it will replace with the check.
       /
      auto latchTerminator = cloneLatch->getTerminator();
      latchTerminator->eraseFromParent();
      IRBuilder<> latchBuilder(cloneLatch);

      /
       * Fetch the value of the loop governing IV that would have been used to
       * check whether the previous iteration was the last one. To do so, we
    need
       * to fetch the value of the loop-governing IV updated by the current
       * iteration, which could be the IV value after updating it by adding the
       * chunking size. So for example, if
       * - the current core excuted the iterations 0, 1, and 2 and
       * - the chunking size is 3 and
       * - there are 2 cores, then
       * at the end of the iteration 2 (i.e., at the latch) of core 0 the
    updated
       * loop-governing IV is 2 (the current value used in the compare
       * instruction)
       * + 1 (the normal IV increment)
       * + 3 (the chunking size) * (2 - 1) (the other cores)
       * ----
       *   6
       *
       * The problem is that we don't know if the header of the iteration 6
    should
       * be executed at all as the loop might have ended at an earlier iteration
       * (e.g., 4). So we need to check whether the previous iteration (5 in the
       * example) was actually executed. To this end, we need to compare the
       * previous iteration IV value (e.g., 5) against the exit condition.
       *
       * Fetch the updated loop-governing IV (6 in the example above).
       /
      auto currentIVValue =
          cast<PHINode>(loopGoverningPHI)->getIncomingValueForBlock(cloneLatch);

      /
       * Compute the value that this IV had at the iteration before (5 in the
       * example above).
       /
      auto prevIterationValue =
          ivUtility
              .generateCodeToComputePreviousValueUsedToCompareAgainstExitConditionValue(
                  latchBuilder,
                  currentIVValue,
                  stepSize);

      /
       * Compare the previous-iteration IV value against the exit condition
       /
      auto clonedCmpInst = updatedCmpInst->clone();
      clonedCmpInst->replaceUsesOfWith(
          valueUsedToCompareAgainstExitConditionValue,
          prevIterationValue);
      latchBuilder.Insert(clonedCmpInst);
      latchBuilder.CreateCondBr(
          clonedCmpInst,
          task->getLastBlock(0),
          headerClone); // the big lesson here is that if iteration n-1 is
    final,
                        // we can't execute the header again. But if n is the
                        // final we MUST execute the header again.
    }
    */

  /*
   * In the preheader, assert that either the first iteration is being executed
   * OR that the previous iteration would have executed. The reason we must also
   * check if this is the first iteration is if the IV condition is such that <=
   * 1 iteration would ever occur
   */

  auto preheaderTerminator = preheaderClone->getTerminator();
  preheaderTerminator->eraseFromParent();
  IRBuilder<> preheaderBuilder(preheaderClone);
  auto offsetStartValue =
      cast<PHINode>(loopGoverningPHI)->getIncomingValueForBlock(preheaderClone);
  auto prevIterationValue =
      ivUtility.generateCodeToComputeValueToUseForAnIterationAgo(
          preheaderBuilder,
          offsetStartValue,
          stepSize);

  // auto clonedExitCmpInst = updatedCmpInst->clone();cmpInst
  // Task 2
  auto clonedExitCmpInst = cmpInst->clone();
  clonedExitCmpInst->replaceUsesOfWith(
      valueUsedToCompareAgainstExitConditionValue,
      prevIterationValue);
  preheaderBuilder.Insert(clonedExitCmpInst);

  auto startValue =
      this->fetchCloneInTask(task, loopGoverningIV->getStartValue());
  assert(startValue != nullptr);
  auto isNotFirstIteration =
      preheaderBuilder.CreateICmpNE(offsetStartValue, startValue);
  preheaderBuilder.CreateCondBr(
      preheaderBuilder.CreateAnd(isNotFirstIteration,
                                 clonedExitCmpInst,
                                 "notFirstIterAndPrevIterOk"),
      task->getExit(),
      headerClone);
}

} // namespace arcana::gino
