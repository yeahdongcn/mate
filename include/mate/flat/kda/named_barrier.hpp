#pragma once

#include <cstdint>

#include "mutlass/arch/barrier.hpp"

namespace mate::flat::kda {

using KdaNamedBarrier = uint32_t;

enum class PrepareBarrier : KdaNamedBarrier {
  GateTotalsReady = 0,
  WorkspaceReady,
  OperandsReady,
  InverseSmemReady,
  WorkspaceStoreDone,
  NumBarriers,
};

enum class RecurrenceBarrier : KdaNamedBarrier {
  UReady = 0,
  StateCommitted,
  ResidualReady,
  NumBarriers,
};

MUTLASS_DEVICE
static uint32_t named_barrier_id(uint32_t barrier_id_) {
  return barrier_id_ + mutlass::arch::AsyncBarrier::ReservedAsyncBarrierCount;
}

template <typename Barrier>
MUTLASS_DEVICE static uint32_t named_barrier_id(Barrier barrier_id_) {
  return named_barrier_id(static_cast<KdaNamedBarrier>(barrier_id_));
}

MUTLASS_DEVICE
static void named_barrier_init(uint32_t barrier_id_, uint32_t arrive_count, uint32_t init_phase = 0) {
  mutlass::arch::AsyncBarrier::init(named_barrier_id(barrier_id_), arrive_count, init_phase);
}

MUTLASS_DEVICE
static void named_barrier_arrive(uint32_t barrier_id_) {
  mutlass::arch::AsyncBarrier::arrive(named_barrier_id(barrier_id_));
}

template <typename Barrier>
MUTLASS_DEVICE static void named_barrier_arrive(Barrier barrier_id_) {
  named_barrier_arrive(static_cast<KdaNamedBarrier>(barrier_id_));
}

MUTLASS_DEVICE
static uint32_t named_barrier_arrive_phase(uint32_t barrier_id_) {
  return mutlass::arch::AsyncBarrier::arrive<true>(named_barrier_id(barrier_id_));
}

template <typename Barrier>
MUTLASS_DEVICE static uint32_t named_barrier_arrive_phase(Barrier barrier_id_) {
  return named_barrier_arrive_phase(static_cast<KdaNamedBarrier>(barrier_id_));
}

MUTLASS_DEVICE
static void named_barrier_wait(uint32_t barrier_id_, uint32_t phase) {
  mutlass::arch::AsyncBarrier::wait(named_barrier_id(barrier_id_), phase);
}

template <typename Barrier>
MUTLASS_DEVICE static void named_barrier_wait(Barrier barrier_id_, uint32_t phase) {
  named_barrier_wait(static_cast<KdaNamedBarrier>(barrier_id_), phase);
}

MUTLASS_DEVICE
static void named_barrier_arrive_and_wait(uint32_t barrier_id_) {
  uint32_t barrier_id = named_barrier_id(barrier_id_);
  uint32_t phase      = mutlass::arch::AsyncBarrier::arrive<true>(barrier_id);
  mutlass::arch::AsyncBarrier::wait(barrier_id, phase);
}

template <typename Barrier>
MUTLASS_DEVICE static void named_barrier_arrive_and_wait(Barrier barrier_id_) {
  named_barrier_arrive_and_wait(static_cast<KdaNamedBarrier>(barrier_id_));
}

}  // namespace mate::flat::kda
