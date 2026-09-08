#pragma once

#include <type_traits>

namespace mate::attention::msa {

template <auto TagValue, typename Default, typename... Options>
struct find_option;

template <auto TagValue, typename Default>
struct find_option<TagValue, Default> {
  using option_value = Default;
};

template <auto TagValue, typename Default, typename Option, typename... Options>
struct find_option<TagValue, Default, Option, Options...>
    : std::conditional_t<Option::tag == TagValue, Option, find_option<TagValue, Default, Options...>> {};

template <auto TagValue, typename Default, typename... Options>
using find_option_t = typename find_option<TagValue, Default, Options...>::option_value;

template <auto TagValue, class Value>
struct Option {
  static constexpr auto tag = TagValue;
  using option_value        = Value;
};

enum class PageTableKind {
  Dense,
  Batched2D,
  Flat,
};

enum class Tag {
  HasMetadata,
  HasSeqUsedK,
  HasCuseqlensQ,
  IsVarlen,
  IsPagedKV,
  IsSparseKV,
  IsCausal,
  IsPackGQA,
  IsSplit,
  UseLSULoadK,
  UseLSULoadV,
  PackQueryPair,
  ParallelKTiles,
  PageSize,
  PageTable,
  MmaTileQ,
  QStages,
  KStages,
  EnableKPrefetch,
};

template <class Value>
using HasMetadata = Option<Tag::HasMetadata, Value>;

template <class Value>
using HasSeqUsedK = Option<Tag::HasSeqUsedK, Value>;

template <class Value>
using HasCuseqlensQ = Option<Tag::HasCuseqlensQ, Value>;

template <class Value>
using IsVarlen = Option<Tag::IsVarlen, Value>;

template <class Value>
using IsPagedKV = Option<Tag::IsPagedKV, Value>;

template <class Value>
using IsSparseKV = Option<Tag::IsSparseKV, Value>;

template <class Value>
using IsCausal = Option<Tag::IsCausal, Value>;

template <class Value>
using IsPackGQA = Option<Tag::IsPackGQA, Value>;

template <class Value>
using IsSplit = Option<Tag::IsSplit, Value>;

template <class Value>
using UseLSULoadK = Option<Tag::UseLSULoadK, Value>;

template <class Value>
using UseLSULoadV = Option<Tag::UseLSULoadV, Value>;

template <class Value>
using PackQueryPair = Option<Tag::PackQueryPair, Value>;

template <class Value>
using ParallelKTiles = Option<Tag::ParallelKTiles, Value>;

template <int Value>
using PageSize = Option<Tag::PageSize, std::integral_constant<int, Value>>;

template <PageTableKind Value>
using PageTable = Option<Tag::PageTable, std::integral_constant<PageTableKind, Value>>;

template <int Value>
using MmaTileQ = Option<Tag::MmaTileQ, std::integral_constant<int, Value>>;

template <int Value>
using QStages = Option<Tag::QStages, std::integral_constant<int, Value>>;

template <int Value>
using KStages = Option<Tag::KStages, std::integral_constant<int, Value>>;

template <bool Value>
using EnableKPrefetch = Option<Tag::EnableKPrefetch, std::bool_constant<Value>>;

}  // namespace mate::attention::msa
