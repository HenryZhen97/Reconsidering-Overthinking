from typing import Callable, Deque, Dict, Iterable, List, Optional
from typing import Sequence as GenericSequence
from typing import Set, Tuple, Union

from vllm.logger import init_logger
from vllm.core.scheduler import Scheduler
from vllm.sequence import SequenceGroup, SequenceStatus

logger = init_logger(__name__)


class Scheduler(Scheduler):
    
    def abort_seq_group(
        self,
        request_id: Union[str, Iterable[str]],
        seq_id_to_seq_group = None,
    ) -> None:
        """Aborts a sequence group with the given ID.
        Check if the sequence group with the given ID
            is present in any of the state queue.
        If present, remove the sequence group from the state queue.
            Also, if any of the sequences in the sequence group is not finished,
                free the sequence with status `FINISHED_ABORTED`.
        Otherwise, do nothing.
        Args:
            request_id: The ID(s) of the sequence group to abort.
            seq_id_to_seq_group: helper for groups with n>1
        """
        if isinstance(request_id, str):
            request_id = (request_id, )
        request_ids = set(request_id)
        seq_id_to_seq_group = seq_id_to_seq_group or {}
        for state_queue in [self.waiting, self.running, self.swapped]:
            aborted_groups: List[SequenceGroup] = []
            for seq_group in state_queue:
                # When n>1, seq_group.request_id looks like
                # foo_parallel_sample_0, while request_ids is just foo, and we
                # should resolve it as real_request_id to match.
                if seq_group.request_id in seq_id_to_seq_group:
                    real_request_id = seq_id_to_seq_group[
                        seq_group.request_id].group_id
                else:
                    real_request_id = seq_group.request_id
                if real_request_id in request_ids:
                    # Appending aborted group into pending list.
                    aborted_groups.append(seq_group)
                    # We can't remove real_request_id in request_ids here,
                    # because there may be other seq groups sharing the same
                    # real_request_id
            for aborted_group in aborted_groups:
                # Remove the sequence group from the state queue.
                state_queue.remove(aborted_group)
                # Remove the aborted request from the Mamba cache.
                self._finished_requests_ids.append(aborted_group.request_id)
                for seq in aborted_group.get_seqs():
                    if seq.is_finished():
                        continue
                    seq.status = SequenceStatus.FINISHED_ABORTED
                    self.free_seq(seq)
                if aborted_group.request_id in seq_id_to_seq_group:
                    del seq_id_to_seq_group[aborted_group.request_id]

                self._free_seq_group_cross_attn_blocks(aborted_group)
