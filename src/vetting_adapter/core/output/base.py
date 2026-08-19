"""Base classes and abstract base classes for creating and writing output.

Classes
-------
ResultOutput
    Abstract base class for creating output from results. Is intended to be take
    results from one instance or a collection of instances of either
    `pathways_ensemble_analysis.Criterion` (including
    `iamcompact_vetting.criteria.TimeseriesRefCriterion`) or
    `iamcompact_vetting.targets.CriterionTargetRange` subclasses. Different
    subclasses will implement output for different types and collections of
    these classes. The subclasses should generally implement the
    `prepare_output` method to return a nominally format-independent output data
    structure, and use a `ResultsWriter` subclass instance to write the output
    to a desired format, through the `write` method.
ResultsWriter
    Abstract base class for writing output. Subclasses should implement writing
    to different formats. In most cases, each non-abstract subclass will be
    intended for use with a specific subclass of `ResultOutput` (which in turn
    is designed to be used with a specific instance or collection of instances
    of `pathways_ensemble_analysis.Criterion` or
    `iamcompact_vetting.targets.CriterionTargetRange` subclasses).
"""
from abc import ABC, abstractmethod
from collections.abc import (
    Callable,
    Mapping,
    Sequence,
)
import copy
import typing as tp
import warnings

import pandas as pd
from pandas.io.formats.style import (
    Styler as PandasStyler,
)
from pandas.io.formats.style_render import (
    Subset as PandasSubset,
)
import pyam

from .column_names import (
    CTCol,
    SummaryColumnSource,
)
from .styling.base import (
    CriterionTargetRangeOutputStyles,
    InRangeStyles,
    PassFailStyles,
)
from ..targets.target_classes import CriterionTargetRange

CritTypeVar = tp.TypeVar('CritTypeVar')
"""TypeVar for the type of `Criterion` or `CriterionTargetRange` expected by a
`ResultOutput` subclass."""
CriterionTargetRangeTypeVar = tp.TypeVar(
    'CriterionTargetRangeTypeVar',
    bound=CriterionTargetRange,
)
OutputDataTypeVar = tp.TypeVar('OutputDataTypeVar')
"""TypeVar for the type of data to be output by a `ResultOutput` subclass
through its `prepare_output` method, and to be written by a `ResultsWriter`
through its `write` method."""
WriteReturnTypeVar = tp.TypeVar('WriteReturnTypeVar')
"""TypeVar for the datatype to be returned by the `write` method of a
`ResultsWriter` subclass."""
StyleTypeVar = tp.TypeVar('StyleTypeVar')
"""TypeVar for style classes."""
StyledOutputTypeVar = tp.TypeVar('StyledOutputTypeVar')
"""TypeVar for output that has been styled."""


class ResultsWriter(ABC, tp.Generic[OutputDataTypeVar, WriteReturnTypeVar]):
    @abstractmethod
    def write(self, output_data: OutputDataTypeVar) -> WriteReturnTypeVar:
        """Write the output data to a desired format.

        Parameters
        ----------
        output_data : OutputDataTypeVar
            The data to be written, prepared into the proper data type by a
            `ResultOutput` subclass instance.

        Returns
        -------
        WriteReturnTypeVar
            Appropriate return value for the `write` method defined by a
            subclass. Can be `None` if no return value is desired.
        """
        ...
#END abstract class ResultsWriter


class NoWriterError(Exception):
    """Raised when the `.write` method of a `NoWriter` instance is called."""
    pass
###END abstract class ResultsWriter

class NoWriter(ResultsWriter[tp.Any, None]):
    """ResultsWriter that can be used when no writer is wanted.

    If the `.write` method of this class is called, a `NoWriterError` is
    raised.

    Init Parameters
    ---------------
    message : str
        Message to be printed when the `.write` method is called.
    """
    def __init__(
            self,
            *,
            message: str = 'No writer has been set for this output.',
    ):
        self.message: str = message
    ###END def NoWriter.__init__
    def write(self, output_data: tp.Any) -> None:
        raise NoWriterError(self.message)
    ###END def NoWriter.write
###END class NoWriter

WriterTypeVar = tp.TypeVar('WriterTypeVar', bound=ResultsWriter)
"""TypeVar for the type of `ResultsWriter` subclass to be used by a
`ResultOutput` subclass instance."""

ResultsInputData: tp.TypeAlias = pyam.IamDataFrame | pd.Series
"""Type alias for the type of data that can be used as input to the
`prepare_output` method of a `ResultOutput` class. Currently set to the union
of `pyam.IamDataFrame` and `pandas.Series`. This is the most general type.
Subclasses may be more restrictive."""

ResultsInputDataTypeVar = tp.TypeVar(
    'ResultsInputDataTypeVar',
    bound=ResultsInputData,
)


class StylingFunc(tp.Protocol):
    """Protocol for methods in that style output DataFrames or Series"""
    def __call__(
        self,
        styler: PandasStyler,
        subset: tp.Optional[PandasSubset] = None,
        *args,
        **kwargs,
    ) -> PandasStyler:
        ...
#END Protocol class StylingFunc


class ResultOutput(
        ABC,
        tp.Generic[
            CritTypeVar,
            ResultsInputDataTypeVar,
            OutputDataTypeVar,
            WriterTypeVar,
            WriteReturnTypeVar
        ]
):
    """Abstract base class for creating output from results.

    Is intended to be take results from one instance or a collection of
    instances of either `pathways_ensemble_analysis.Criterion` (including
    `iamcompact_vetting.criteria.TimeseriesRefCriterion`) or
    `iamcompact_vetting.targets.CriterionTargetRange` subclasses. Different
    subclasses will implement output for different types and collections of
    these classes. The subclasses should generally implement the
    `prepare_output` method to return a nominally format-independent output
    data structure, and use a `ResultsWriter` subclass instance to write the
    output to a desired format, through the `write` method.
    """

    def __init__(
            self,
            *,
            criteria: CritTypeVar,
            writer: WriterTypeVar,
    ) -> None:
        """
        All parameters are keyword-only, to allow for future changes in the
        number of input parameters.

        Parameters
        ----------
        criteria : CritTypeVar
            The criteria that will be used to produce results that are to be
            output by the `ResultOutput` instance.
        writer : WriterTypeVar
            The writer to be used to write the output.
        """
        self.criteria: CritTypeVar = criteria
        self.writer: WriterTypeVar= \
            writer
    ###END def ResultOutput.__init__

    @abstractmethod
    def prepare_output(
            self,
            data: ResultsInputDataTypeVar,
            /,
            criteria: tp.Optional[CritTypeVar] = None,
            **kwargs
    ) -> OutputDataTypeVar:
        """Prepare the output data for writing.

        The method takes data (in the form of a `pyam.IamDataFrame` or
        `pandas.Series`, or similar data structure required by a given
        subclass), passes it to the instance or instances store in `criteria`,
        and prepares the proper output data structure for writing based on the
        results.

        Parameters
        ----------
        data : ResultsInputDataTypeVar
            The data to be passed to `criteria` before the results are prepared
            into output.
        critera : CritTypeVar, optional
            The criteria instance(s) to be used for processing `data` and
            producing results to be prepared for output. If `None` (default),
            `self.criteria` should be used (which was set by the `__init__`
            method), but subclasses have to implement this. The method in the
            base class is an empty abstract method.
        **kwargs
            Additional keyword arguments that can be used by subclasses.

        Returns
        -------
        OutputDataTypeVar
            The data to be written, prepared into the proper data type by a
            `ResultOutput` subclass instance.
        """
        ...
    ###END def ResultOutput.prepare_output

    def write_output(
            self,
            output: OutputDataTypeVar,
            /,
            writer: tp.Optional[WriterTypeVar] \
                = None,
            **kwargs
    ) -> WriteReturnTypeVar:
        """Write the output data to the format written by `writer`.

        This method is used for result outputs that have already been prepared
        into the proper format, usually by `self.prepare_output`. To write
        outputs directly based on `Criterion` or `CriterionTargetRange`
        instances, use `self.write_rsults` instead.

        Parameters
        ----------
        output : OutputDataTypeVar
            The data to be written, prepared into the proper output data
            structure, usually through `self.prepare_output`.
        writer : tp.Optional[WriterTypeVar]
            The writer to be used to write the output. If `None`, the
            `self.writer` attribute is used.
        **kwargs
            Additional keyword arguments to be passed to `writer.write`.

        Returns
        -------
        WriteReturnTypeVar
            The return value from the `writer.write` method.
        """
        if writer is None:
            writer = self.writer
        return writer.write(output, **kwargs)
    ###END def ResultOutput.write_output

    def write_results(
            self,
            data: ResultsInputDataTypeVar,
            /,
            criteria: tp.Optional[CritTypeVar] = None,
            *,
            writer: tp.Optional[
                WriterTypeVar
            ] = None,
            prepare_output_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
            write_output_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
    ) -> tuple[OutputDataTypeVar, WriteReturnTypeVar]:
        """Write the results to the format written by `writer`.

        Parameters
        ----------
        results : CritTypeVar
            The results to be prepared into the proper data structure through
            `self.prepare_output`, which is then written through
            `self.write_output`, using the `writer` parameter.
        writer : tp.Optional[WriterTypeVar]
            The writer to be used to write the output. If `None`, the
            `self.writer` attribute is used.
        prepare_output_kwargs : dict, optional
            Additional keyword arguments to be passed to `prpare_output`.
        write_output_kwargs : dict, optional
            Additional keyword arguments to be passed to `write_output`.

        Returns
        -------
        WriteReturnTypeVar
            The return value from the `writer.write` method.
        """
        if prepare_output_kwargs is None:
            prepare_output_kwargs = dict()
        if write_output_kwargs is None:
            write_output_kwargs = dict()
        if criteria is None:
            criteria = self.criteria
        output: OutputDataTypeVar = self.prepare_output(
            data,
            criteria=criteria,
            **prepare_output_kwargs,
        )
        write_returnval: WriteReturnTypeVar = \
            self.write_output(output, writer, **write_output_kwargs)
        return (output, write_returnval)
    ###END def ResultOutput.write_results

    def set_writer(
            self,
            writer: WriterTypeVar
    ) -> None:
        """Set a new writer for the instance.

        Parameters
        ----------
        writer : WriterTypeVar
            The new writer to be used.

        Returns
        -------
        None
            No value is returned. If you want to set a writer as part of a
            pipeline and return `self`, use `self.with_writer` instead, with
            `inplace` set to `True`.
        """
        self.writer = writer
    ###END def ResultOutput.set_writer

    def with_writer(
            self,
            writer: WriterTypeVar,
            inplace: bool = False,
    ) -> tp.Self:
        """Return a copy or the instance itself with a new writer object set.

        Note that only a shallow copy is returned. The criteria objects and
        other attributes still point to the same objects.

        Parameters
        ----------
        writer : WriterTypeVar
            The new writer to be used.

        Returns
        -------
        A copy with a new writer set, or `self` with a new writer set if
        `inplace` is `True`.
        """
        output_obj: tp.Self = self if inplace else copy.copy(self)
        output_obj.set_writer(writer)
        return output_obj
    ###END def ResultOutput.with_writer

###END abstract class ResultOutput


class StyledResultOutput(
        ResultOutput[
            CritTypeVar,
            ResultsInputDataTypeVar,
            OutputDataTypeVar,
            WriterTypeVar,
            WriteReturnTypeVar
        ],
        tp.Generic[
            CritTypeVar,
            ResultsInputDataTypeVar,
            OutputDataTypeVar,
            WriterTypeVar,
            WriteReturnTypeVar,
            StyleTypeVar,
            StyledOutputTypeVar
        ]
):
    """ResultsOutput class with styling support.

    The class adds parameters for specifying whether styling should be used in
    the `prepare_output` and `write_output` methods, and to set a styler object
    as an instance attribute. Actual styling has to implemented by subclasses,
    by specifying the class of styler objects, and by implementing the
    `style_output` method.
    """

    def __init__(
            self,
            *,
            criteria: CritTypeVar,
            writer: WriterTypeVar,
            style: StyleTypeVar,
    ) -> None:
        """
        All the parameters are keyword-only, to allow for future changes in the
        number of input parameters.

        Parameters
        ----------
        criteria : CritTypeVar
            The criteria that will be used to produce results that are to be
            output by the `ResultOutput` instance.
        writer : WriterTypeVar
            The writer to be used to write the output.
        styler : StylerTypeVar
            The styler object to be used to style the output. The type of object
            and its behavior must be determined by subclasses.
        """
        super().__init__(criteria=criteria, writer=writer)
        self.style: StyleTypeVar = style
    ###END def StyledResultOutput.__init__

    @abstractmethod
    def style_output(
            self,
            output: OutputDataTypeVar,
    ) -> StyledOutputTypeVar:
        """Style output prepared by `self.prepare_output`.

        Abstract method, must be implemented by subclasses.
        """
        raise NotImplementedError
    ###END abstractmethod def StyledResultOutput.style_output

    def prepare_styled_output(
            self,
            data: ResultsInputDataTypeVar,
            *,
            prepare_output_kwargs: tp.Optional[dict[str, tp.Any]] = None,
            style_output_kwargs: tp.Optional[dict[str, tp.Any]] = None,
            **kwargs,
    ) -> StyledOutputTypeVar:
        """Prepare output and style it.

        Calls `self.prepare_output` on the input data, and then
        `self.style_output` on the output.

        Parameters
        ----------
        data : ResultsInputDataTypeVar
            The data to be used to prepare the output.
        prepare_output_kwargs : dict, optional
            Additional keyword arguments to be passed to `prepare_output`.
        style_output_kwargs : dict, optional
            Additional keyword arguments to be passed to `style_output`.
        **kwargs
            Additional keyword arguments are passed to both `prepare_output`
            and `style_output`. They will be overridden by keywords that are
            present in `prepare_output_kwargs` or `style_output_kwargs`.

        Returns
        -------
        StyledOutputTypeVar
            The styled output.
        """
        if prepare_output_kwargs is None:
            prepare_output_kwargs = dict()
        if style_output_kwargs is None:
            style_output_kwargs = dict()
        output: OutputDataTypeVar = self.prepare_output(
            data,
            **(kwargs | prepare_output_kwargs),
        )
        styled_output: StyledOutputTypeVar = self.style_output(
            output,
            **(kwargs | style_output_kwargs),
        )
        return styled_output
    ###END def StyledResultOutput.prepare_styled_output

    def write_output(
            self,
            output: OutputDataTypeVar|StyledOutputTypeVar,
            /,
            writer: tp.Optional[WriterTypeVar] = None,
            **kwargs,
    ) -> WriteReturnTypeVar:
        """Write output data to the format written by `writer`.

        This method is used for result outputs that have already been prepared
        into the proper format, and possibly styled, usually by
        `self.prepare_output` or `self.prepare_styled_output`. To write outputs
        directly based on input data, use `self.write_results`.

        The parameter `output` is assumed to be either unstyled or styled
        output, but `writer` must support whichever type is used.

        Parameters
        ----------
        output : OutputDataTypeVar or StyledOutputTypeVar
            The data to be written, prepared into the proper output data
            structure, usually through `self.prepare_output` or
            `self.prepare_styled_output`.
        writer : tp.Optional[WriterTypeVar]
            The writer to be used to write the output. If `None`, the
            `self.writer` attribute is used.
        **kwargs
            Additional keyword arguments to be passed to `writer.write`.

        Returns
        -------
        WriteReturnTypeVar
            The return value of `writer.write`.
        """
        if writer is None:
            writer = self.writer
        return writer.write(output, **kwargs)
    ###END def StyledResultOutput.write_output

    def write_results(
            self,
            data: ResultsInputDataTypeVar,
            /,
            criteria: tp.Optional[CritTypeVar] = None,
            *,
            writer: tp.Optional[WriterTypeVar] = None,
            style_output: tp.Optional[bool] = None,
            prepare_output_kwargs: tp.Optional[dict[str, tp.Any]] = None,
            style_output_kwargs: tp.Optional[dict[str, tp.Any]] = None,
            write_output_kwargs: tp.Optional[dict[str, tp.Any]] = None,
    ) -> tuple[OutputDataTypeVar|StyledOutputTypeVar, WriteReturnTypeVar]:
        """See the documentation for `ResultsOutput.write_results`.

        Output is styled before writing if `style_output` is `True`. The
        parameters of `style_output_kwargs` are passed as keyword arguments
        to `self.style_output`. Apart from this, the behavior and parameters
        are the same as for `ResultsOutput.write_results`.
        """
        if style_output is None:
            style_output = True
        if not style_output:
            return super().write_results(
                data,
                criteria=criteria,
                writer=writer,
                prepare_output_kwargs=prepare_output_kwargs,
                write_output_kwargs=write_output_kwargs,
            )
        if prepare_output_kwargs is None:
            prepare_output_kwargs = dict()
        if style_output_kwargs is None:
            style_output_kwargs = dict()
        if write_output_kwargs is None:
            write_output_kwargs = dict()
        if criteria is None:
            criteria = self.criteria
        styled_output: StyledOutputTypeVar = self.prepare_styled_output(
            data,
            prepare_output_kwargs=prepare_output_kwargs,
            style_output_kwargs=style_output_kwargs,
        )
        write_returnval: WriteReturnTypeVar = self.write_output(
            styled_output,
            writer=writer,
            **write_output_kwargs,
        )
        return (styled_output, write_returnval)
    ###END def StyledResultOutput.write_results

###END class StyledResultOutput



class CriterionTargetRangeOutput(
        StyledResultOutput[
            CriterionTargetRangeTypeVar,
            pyam.IamDataFrame,
            pd.DataFrame,
            WriterTypeVar,
            WriteReturnTypeVar,
            CriterionTargetRangeOutputStyles,
            PandasStyler
        ],
):
    """TODO: NEED TO ADD PROPER DOCSTRING"""

    _default_columns: list[CTCol]
    _default_column_titles: Mapping[CTCol, str]

    def __init__(
            self,
            *,
            criteria: CriterionTargetRangeTypeVar,
            writer: WriterTypeVar,
            columns: tp.Optional[Sequence[CTCol]] = None,
            column_titles: tp.Optional[Mapping[CTCol, str]] = None,
            style: tp.Optional[CriterionTargetRangeOutputStyles] = None,
    ):
        """TODO: NEED TO ADD PROPER DOCSTRING"""
        super().__init__(
            criteria=criteria,
            writer=writer,
            style=style if style is not None \
                else CriterionTargetRangeOutputStyles(),
        )
        if columns is not None:
            self._default_columns = list(columns)
        elif not hasattr(self, '_default_columns'):
            self._default_columns = [
                CTCol.INRANGE,
                CTCol.DISTANCE,
                CTCol.VALUE,
            ]
        if column_titles is not None:
            self._default_column_titles = column_titles
        elif not hasattr(self, '_default_column_titles'):
            self._default_column_titles = {
                CTCol.INRANGE: 'Is in target range',
                CTCol.DISTANCE: 'Rel. distance from target',
                CTCol.VALUE: 'Value',
            }
        self.styling_funcs: tp.Final[dict[CTCol, StylingFunc]] = {
            CTCol.INRANGE: self.style_in_range_columns,
            CTCol.VALUE: self.style_value_columns,
            CTCol.DISTANCE: self.style_distance_columns,
        }

    ###END def CriterionTargetRangeOutput.__init__

    def prepare_output(
            self,
            data: pyam.IamDataFrame,
            /,
            criteria: tp.Optional[CriterionTargetRangeTypeVar] = None,
            *,
            columns: tp.Optional[Sequence[CTCol]] = None,
            column_titles: tp.Optional[Mapping[CTCol, str]] = None,
    ) -> pd.DataFrame:
        """TODO: NEED TO ADD PROPER DOCSTRING"""
        if columns is None:
            columns = self._default_columns
        if column_titles is None:
            column_titles = self._default_column_titles
        if criteria is None:
            criteria = self.criteria
        result_columns: list[pd.Series] = []
        for _col in columns:
            if _col == CTCol.INRANGE:
                result_columns.append(
                    criteria.get_in_range(data).rename(_col)
                )
            elif _col == CTCol.DISTANCE:
                result_columns.append(
                    criteria.get_distances(data).rename(_col)
                )
            elif _col == CTCol.VALUE:
                result_columns.append(
                    criteria.get_values(data).rename(_col)
                )
            else:
                raise ValueError(f'Unrecognized column: {_col!r}')
        results_df: pd.DataFrame = pd.concat(result_columns, axis=1)
        if column_titles is not None:
            results_df = results_df.rename(columns=column_titles)
        return results_df
    ###END def CriterionTargetRangeOutput.prepare_output

    def style_output(
            self,
            output: pd.DataFrame,
            *,
            column_titles: tp.Optional[Mapping[CTCol, str]] = None,
    ) -> PandasStyler:
        """Apply styles to the output.

        The styles to apply to the different columns are taken from
        `self.style`. To use a different style than the one that was specified
        during intialization, set the `self.style` attribute.

        Parameters
        ----------
        output : pd.DataFrame
            The output to be styled.
        column_titles : Mapping[CTCol, str], optional
            A mapping from `CTCol` to the name of the columns in `output`. This
            parameter is used to decide what styles to apply to which columns.
            Optional, by default uses `self._default_column_titles`.
        """
        if column_titles is None:
            column_titles = self._default_column_titles
        if not set(output.columns).issubset(set(column_titles.values())):
            raise ValueError(
                'The following columns are not found in `column_titles`, and '
                'cannot be styled: '
                f'{set(output.columns) - set(column_titles.values())}'
            )
        return_style: PandasStyler = output.style
        self.apply_common_styling(return_style)
        for _col in column_titles:
            return_style = self.styling_funcs[_col](
                return_style,
                subset=[column_titles[_col]],
            )
        return return_style
    ###END def CriterionTargetRangeOutput.style_output

    def apply_common_styling(
            self,
            styler: PandasStyler,
            *,
            style: tp.Optional[CriterionTargetRangeOutputStyles] = None,
    ) -> PandasStyler:
        """Apply common (not column-specific) styling.

        Applies common styles, such as common number formatting or styling of
        indexes and column headers.

        Parameters
        ----------
        styler : pandas Styler
            The pandas Styler to be styled.
        style : CriterionTargetRangeOutputStyles, optional
            The object with styling information to apply.

        Returns
        -------
        styler : pandas Styler
            `styler` with styling applied
        """
        if style is None:
            style = self.style
        if style.common is not None:
            styler = styler.format(**style.common.FORMAT)
            if style.common.INDEX_FORMAT is not None:
                styler = \
                    styler.format_index(**style.common.INDEX_FORMAT, axis=0)
            if style.common.COLUMN_FORMAT is not None:
                styler = \
                    styler.format_index(**style.common.COLUMN_FORMAT, axis=1)
            if style.common.INDEX_STYLE is not None:
                index_style: Callable[[tp.Any], str] | str = \
                    style.common.INDEX_STYLE
                if callable(index_style):
                    styler = styler.map_index(index_style, axis=0)
                else:
                    styler = styler.map_index(lambda x: index_style, axis=0)
            if style.common.COLUMN_STYLE is not None:
                column_style: Callable[[tp.Any], str] | str = \
                    style.common.COLUMN_STYLE
                if callable(column_style):
                    styler = styler.map_index(column_style, axis=1)
                else:
                    styler = styler.map_index(lambda x: column_style, axis=1)
        return styler
    ###END def CriterionTargetRangeOutput.apply_common_styling

    def style_in_range_columns(
            self,
            styler: PandasStyler,
            subset: tp.Optional[PandasSubset] = None,
            *,
            style: tp.Optional[CriterionTargetRangeOutputStyles] = None,
    ) -> PandasStyler:
        """Style columns that contain a True/False value for in-range status.

        Parameters
        ----------
        styler : pandas Styler
            The pandas Styler to be styled.
        subset : str, array-like, or pandas.IndexSlice
            Indexer for the subset of the Styler's DataFrame to which the
            styling is applied. See, e.g., `Styler.format` in the pandas
            documentation for more details. To style one or more columns, you
            can pass a list of strings.
        style : CriterionTargetRangeOutputStyles, optional
            The object with styling information to apply.

        Returns
        -------
        styler : pandas Styler
            `styler` with styling applied
        """
        if style is None:
            style = self.style
        subset_styles: PassFailStyles = style.in_range
        styler = styler.format(
            subset=subset,
            **(subset_styles.FORMAT)
        )
        styler = styler.map(
            func=lambda x: subset_styles.NA if pd.isna(x) \
                else subset_styles.PASS if x \
                else subset_styles.FAIL,
            subset=subset,
        )
        return styler
    ###END def CriterionTargetRangeOutput.style_in_range_columns

    def style_value_columns(
            self,
            styler: PandasStyler,
            subset: tp.Optional[PandasSubset] = None,
            *,
            style: tp.Optional[CriterionTargetRangeOutputStyles] = None,
    ) -> PandasStyler:
        """Style columns with numeric values comparable to a target range.

        Parameters
        ----------
        styler : pandas Styler
            The pandas Styler to be styled.
        subset : str, array-like, or pandas.IndexSlice
            Indexer for the subset of the Styler's DataFrame to which the
            styling is applied. See, e.g., `Styler.format` in the pandas
            documentation for more details. To style one or more columns, you
            can pass a list of strings.
        style : CriterionTargetRangeOutputStyles, optional
            The object with styling information to apply.

        Returns
        -------
        styler : pandas Styler
            `styler` with styling applied
        """
        if style is None:
            style = self.style
        criterion: CriterionTargetRange = self.criteria
        subset_styles: InRangeStyles = style.value
        styler = styler.format(
            subset=subset,
            **(subset_styles.FORMAT)
        )
        if criterion.range is None:
            warnings.warn(
                f'The criterion with name "{criterion.name}" does not have a '
                '`range`. Value columns will not be styled.'
            )
            return styler
        styler = styler.map(
            func=lambda x: subset_styles.NA if pd.isna(x) \
                else subset_styles.IN_RANGE \
                    if criterion.is_in_range(x) == True \
                else subset_styles.BELOW_RANGE \
                    if criterion.is_below_range(x) == True \
                else subset_styles.ABOVE_RANGE \
                    if criterion.is_above_range(x) == True \
                else None,
            subset=subset,
        )
        return styler
    ###END def CriterionTargetRangeOutput.style_value_columns

    def style_distance_columns(
            self,
            styler: PandasStyler,
            subset: tp.Optional[PandasSubset] = None,
            *,
            style: tp.Optional[CriterionTargetRangeOutputStyles] = None,
    ) -> PandasStyler:
        """Style columns with distance values compared to a target range.

        Parameters
        ----------
        styler : pandas Styler
            The pandas Styler to be styled.
        subset : str, array-like, or pandas.IndexSlice
            Indexer for the subset of the Styler's DataFrame to which the
            styling is applied. See, e.g., `Styler.format` in the pandas
            documentation for more details. To style one or more columns, you
            can pass a list of strings.
        style : CriterionTargetRangeOutputStyles, optional
            The object with styling information to apply.

        Returns
        -------
        styler : pandas Styler
            `styler` with styling applied
        """
        if style is None:
            style = self.style
        criterion: CriterionTargetRange = self.criteria
        subset_styles: InRangeStyles = style.distance
        styler = styler.format(
            subset=subset,
            **(subset_styles.FORMAT)
        )
        warnings.warn(
            'Styling for distance columns is not yet implemented for this '
            f'class ({self.__class__.__name__}). Styles other than number '
            'formats will not be applied.'
        )
        return styler
    ###END def CriterionTargetRangeOutput.style_distance_columns

    # def get_target_range_values(
    #         self,
    #         index: pd.MultiIndex,
    #         include_values: tp.Optional[TargetRangeValue] = None,
    #         criteria: tp.Optional[CriterionTargetRangeTypeVar] = None,
    #         *,
    #         columns: tp.Optional[Sequence[CTCol]] = None,
    #         column_titles: tp.Optional[Mapping[CTCol, str]] = None,
    # ) -> pd.DataFrame:
    #     ...

###END class CriterionTargetRangeOutput


DataFrameMappingWriterTypeVar = tp.TypeVar(
    'DataFrameMappingWriterTypeVar',
    bound=ResultsWriter[Mapping[str, pd.DataFrame], tp.Any],
)

CriterionTargetRangeOutputTypeVar = tp.TypeVar(
    'CriterionTargetRangeOutputTypeVar',
    bound=CriterionTargetRangeOutput
)


class MultiCriterionTargetRangeOutput(
        StyledResultOutput[
            Mapping[str, CriterionTargetRangeTypeVar],
            pyam.IamDataFrame,
            dict[str, pd.DataFrame],
            DataFrameMappingWriterTypeVar,
            tp.Any,
            Mapping[str, CriterionTargetRangeOutputStyles|None],
            dict[str, PandasStyler]
        ],
        tp.Generic[
            CriterionTargetRangeTypeVar,
            CriterionTargetRangeOutputTypeVar,
            DataFrameMappingWriterTypeVar
        ]
):
    """Class to make output for multiple CriterionTargetRange instances.

    The class takes a dictionary of `CriterionTargetRangeOutput` instances with
    str names as keys, and returns a dictionary of DataFrames, each produced
    using a `CriterionTargetRange` instance to produce the given DataFrame.

    The dictionary of `CriterionTargetRangeOutput` instances can be subclasses
    of `CriterionTargetRangeOutput`. If so, you should pass the subclass type
    to the `criteria_class` keyword argument of the `__init__` method. The
    implementation of this class assumes that all the
    """

    _default_columns: list[CTCol]
    _default_column_titles: dict[CTCol, str]
    _default_include_summary: bool
    _default_summary_keys: dict[CTCol, str]
    _default_summary_column_name_source: SummaryColumnSource

    class InitKwargsType(tp.TypedDict, total=False):
        columns: tp.Optional[tp.Sequence[CTCol]]
        column_titles: tp.Optional[tp.Dict[CTCol, str]]
    ###END class MultiCriterionTargetRangeOutput.KwargType
    """Type annotations for keyword arguments to the `__init__` method.

    Can be used by subclasses that wish to override the `__init__` method
    without repeating the entire function signature.
    """

    def __init__(
            self,
            *,
            criteria: Mapping[str, CriterionTargetRangeTypeVar],
            writer: DataFrameMappingWriterTypeVar,
            columns: tp.Optional[tp.Sequence[CTCol]] = None,
            column_titles: tp.Optional[tp.Mapping[CTCol, str]] = None,
            include_summary: tp.Optional[bool] = None,
            summary_keys: tp.Optional[tp.Mapping[CTCol, str]] = None,
            criteria_output_class: tp.Type[CriterionTargetRangeOutputTypeVar] \
                = CriterionTargetRangeOutput,
            summary_column_name_source: tp.Optional[SummaryColumnSource] = None,
            style: tp.Optional[CriterionTargetRangeOutputStyles] \
                | Mapping[str, tp.Optional[CriterionTargetRangeOutputStyles]] \
                    = None
    ):
        """TODO: NEED TO ADD PROPER DOCSTRING"""
        self.criteria_output_class: \
            tp.Final[tp.Type[CriterionTargetRangeOutputTypeVar]] \
                = criteria_output_class
        if include_summary is not None:
            self._default_include_summary = include_summary
        else:
            self._default_include_summary = True if summary_keys is not None \
                else False
        if summary_keys is not None:
            self._default_summary_keys = dict(**summary_keys)
        elif not hasattr(self, '_default_summary_keys'):
            self._default_summary_keys = self._default_column_titles.copy()
        if isinstance(style, CriterionTargetRangeOutputStyles) or style is None:
            _singlestyle: CriterionTargetRangeOutputStyles|None = style
            if _singlestyle is None:
                style = {
                    _key: CriterionTargetRangeOutputStyles()
                    for _key in self._default_summary_keys.values()
                }
            else:
                style = {
                    _key: _singlestyle
                    for _key in self._default_summary_keys.values()
                }
            style |= {_key: _singlestyle for _key in criteria.keys()}
        if not isinstance(style, Mapping):
            raise TypeError(
                '`style` must be an instance of '
                '`CriterionTargetRangeOutputStyles`, None, or a Mapping from '
                'string criteria keys to CriterionTargetRangeOutputStyles.'
            )
        super().__init__(criteria=criteria, writer=writer, style=style)
        if columns is not None:
            self._default_columns = list(columns)
        elif not hasattr(self, '_default_columns'):
            self._default_columns = list(columns) if columns is not None else [
                CTCol.INRANGE,
                CTCol.DISTANCE,
                CTCol.VALUE,
            ]
        if column_titles is not None:
            self._default_column_titles = dict(**column_titles)
        elif not hasattr(self, '_default_column_titles'):
            self._default_column_titles = {
                CTCol.INRANGE: 'Is in target range',
                CTCol.DISTANCE: 'Rel. distance from target',
                CTCol.VALUE: 'Value',
            }
        if summary_column_name_source is not None:
            self._default_summary_column_name_source = \
                summary_column_name_source
        elif not hasattr(self, '_default_summary_column_name_source'):
            self._default_summary_column_name_source = \
                SummaryColumnSource.CRITERIA_NAMES
    ###END def MultiCriterionTargetRangeOutput.__init__


    def _get_column_titles(
            self,
            column_titles: Mapping[str, Mapping[CTCol, str]] | \
                Mapping[CTCol, str] | None,
            criteria: Mapping[str, CriterionTargetRangeTypeVar],
    ) -> dict[str, dict[CTCol, str]]:
        """Get a column_titles dict to use, based on input parameters.

        Used by methods that take a possibly optional `column_titles` parameter
        to obtain an appropritate `column_titles` dict to use. Separated out as
        a separate method, since the behavior neeeds to be the same across
        methods, such as `prepare_output` and `prepare_summary_output`.

        Parameters
        ----------
        column_titles : Mapping[str, Mapping[CTCol, str]] | Mapping[CTCol, str] | None
            The column titles input parameter to process.
        criteria : Mapping[str, CriterionTargetRangeTypeVar]
            The `criteria` parameter that was passed to the calling method. Is
            used to determine the keys of the returned dict.

        Returns
        -------
        dict[str, dict[CTCol, str]]
            The column titles dict to use. The dict will have the same keys as
            `criteria`. If `column_titles` is a Mapping of Mappings, a `ValueError`
            will be raised if it does not have the same keys as `criteria`. If
            `column_titles` is a single Mapping from `CTCol` to `str`, the dict
            will turn that mapping into a dict and use it as the value for all
            items in the returned dict. If `column_titles` is None,
            `self._default_column_titles` will be used as the value for all
            items.

        Raises
        ------
        ValueError
            If `column_titles` is a Mapping of Mappings and it does not have the
            same keys as `criteria`.
        """
        if column_titles is None:
            return {_key: self._default_column_titles for _key in criteria}
        elif not isinstance(column_titles, Mapping):
            raise TypeError(
                '`column_titles` must be None or a Mapping, not '
                f'{type(column_titles)}'
            )
        if isinstance(list(column_titles.values())[0], Mapping):
            if set(column_titles.keys()) != set(criteria):
                raise ValueError(
                    '`column_titles` must have the same keys as `criteria`.'
                )
            return dict(**column_titles)
        else:
            column_titles = tp.cast(Mapping[CTCol, str], column_titles)
            return {_key: dict(column_titles) for _key in criteria.keys()}
    ###END def MultiCriterionTargetRangeOutput._get_column_titles

    def prepare_output(
            self,
            data: pyam.IamDataFrame,
            /,
            criteria: tp.Optional[Mapping[str, CriterionTargetRangeTypeVar]] \
                = None,
            *,
            columns: tp.Optional[
                Mapping[str, Sequence[CTCol]] | Sequence[CTCol]
            ] = None,
            column_titles: tp.Optional[Mapping[str, Mapping[CTCol, str]]] \
                = None,
            add_summary_output: tp.Optional[bool] = None,
            columns_in_summary: tp.Optional[Sequence[CTCol]] = None,
            summary_column_name_source: tp.Optional[SummaryColumnSource|str] \
                = None,
            summary_drop_levels: tp.Optional[str|Sequence[str]] = None,
            summary_keys: tp.Optional[Mapping[CTCol, str]] = None,
    ) -> dict[str, pd.DataFrame]:
        """TODO: NEED TO ADD PROPER DOCSTRING

        Note that the kwargs starting with `summary_` are only used if
        `add_summary_output` is `True`, otherwise they are ignored.
        """
        if add_summary_output is None:
            add_summary_output = self._default_include_summary
        if criteria is None:
            criteria = self.criteria
        if columns is None:
            columns = self._default_columns
        column_titles = self._get_column_titles(
            column_titles=column_titles,
            criteria=criteria,
        )
        if add_summary_output:
            if columns_in_summary is None:
                if isinstance(columns, Mapping):
                    raise ValueError(
                        'Received `None` for `columns_in_summary`. If '
                        '`add_summary_output` is `True`, and `columns` is a '
                        'mapping from strings to column names (i.e., the '
                        'columns to include can vary by criterion), '
                        '`columns_in_summary` cannot be inferred from '
                        '`columns` and must be specified.'
                    )
                columns_in_summary = columns
        output_dict: dict[str, pd.DataFrame] = {
            _name: self.criteria_output_class(
                criteria=_criterion,
                writer=self.writer,
                columns=columns[_name] if isinstance(columns, Mapping) \
                    else columns,
                column_titles=column_titles[_name],
            ).prepare_output(data)
            for _name, _criterion in criteria.items()
        }
        if add_summary_output:
            output_dict = self.prepare_summary_output(
                criteria=criteria,
                prepared_output=output_dict,
                columns=columns_in_summary,
                column_titles=column_titles,
                column_name_source=summary_column_name_source,
                drop_levels=summary_drop_levels,
                summary_keys=summary_keys,
            ) | output_dict
        return output_dict
    ###END def MultiCriterionTargetRangeOutput.prepare_output

    def prepare_summary_output(
            self,
            data: tp.Optional[pyam.IamDataFrame] = None,
            /,
            criteria: tp.Optional[Mapping[str, CriterionTargetRangeTypeVar]] \
                = None,
            *,
            prepared_output: tp.Optional[dict[str, pd.DataFrame]] = None,
            columns: tp.Optional[Sequence[CTCol]] = None,
            column_titles: tp.Optional[Mapping[str, Mapping[CTCol, str]]] \
                = None,
            summary_keys: tp.Optional[Mapping[CTCol, str]] = None,
            column_name_source: tp.Optional[SummaryColumnSource|str] = None,
            drop_levels: tp.Optional[str|Sequence[str]] = None,
    ) -> dict[str, pd.DataFrame]:
        """Create DataFrame with values from one column for all criteria.

        Returns a dict of DataFrames, where each DataFrame contains the values
        for a single column from the output from each criterion, and has each
        criterion along the columns.

        **NB!** The current implementation of this method in practice assumes
        that the `.get_values` method of each criterion will return a `Series`
        with the same index levels, including both level names and level
        ordering, after the levels listed in `drop_levels` have been removed. A
        future version might take care to do whatever broadcasting, level
        reordering and reindexing that is necessary to align the Series
        properly. But for now you will need to ensure that all the criteria
        return Series with the same index levels, and override them with
        subclasses if necessary.

        Parameters
        ----------
        data : pyam.IamDataFrame
            The data to be used in the output.
        criteria : Mapping[str, CriterionTargetRangeTypeVar], optional
            The criteria to be used in the output. If `None` (default) and
            `prepared_output` is `None`, `self.criteria` will be used.
        prepared_output : Mapping[str, pd.DataFrame], optional
            Already prepared output from `self.prepare_output`. Can be used to
            boost efficiency, as the method will otherwise call
            `self.prepare_output`. If you specify `prepared_output` you must
            ensure that the `criteria` is identical to criteria passed to the
            original call to `prepare_output` that was used to prepare the
            DataFrames. If `columns` is specified, you must also ensure that it
            is identical. `column_titles` will be ignored.
        columns : Mapping[str, Sequence[CTCol]], optional
            The columns to include summaries for. If `None` (default),
            `self._default_columns` will be used.
        column_titles : Mapping[str, Mapping[CTCol, str]], optional
            The column titles to be used in the output. If `None` (default),
            `self._default_column_titles` will be used. The titles will be the
            keys of the returned dict.
        column_name_source : SummaryColumnSource or str, optional
            Where to get the column names of the summary DataFrames from (which
            identify the criteria). If a str, the names will be taken from an
            index level, which must then be present in the Series returned by
            the `.get_values` method of each criterion in `criteria`. **NB!** If
            you pass a string value for `column_name_source`, you will probably
            also need to pass a value for `drop_levels`, to ensure that the
            specified index level is not dropped, since dropping levels takes
            place before obtaining column names. If `column_name_source` is a
            `SummaryColumnSource` enum, the following will be used:
              * `SummaryColumnSource.DICT_KEYS`: The column names will be
                taken from the dictionary keys of `criteria`.
              * `SummaryColumnSource.CRITERIA_NAMES`: The column names will be
                taken from the `.name` property of each criterion in `criteria`.
            Optional, the deafult is `SummaryColumnSource.CRITERIA_NAMES`.
        drop_levels : str or Sequence[str], optional
            Level names to be dropped from the index of the returned DataFrame
            from each criterion, and not included in the index of the DataFrame
            returned from this method. If `column_name_source` is a string, the
            corresponding level is always dropped (since it is used in the
            column names), and **must be not** included here. Any levels with
            the given names will be dropped, but no error is raised if the given
            names are not present in the indexes of the Series returned by any
            of the criteria in `criteria`. Optional, the default is to drop the
            levels `variable` and the level names given in the
            `.rename_variable_column` attribute of each `CriterionTargetRange`
            instance in `criteria`. If you do not wish to drop any levels, pass
            in an empty tuple or list.
        """
        if criteria is None:
            criteria = self.criteria
        column_titles = self._get_column_titles(
            column_titles=column_titles,
            criteria=criteria,
        )
        if columns is None:
            columns = self._default_columns
        if summary_keys is None:
            summary_keys = self._default_summary_keys
        if column_name_source is None:
            column_name_source = self._default_summary_column_name_source
        if drop_levels is None:
            drop_levels = ('variable',) + tuple(
                _crit.rename_variable_column for _crit in criteria.values()
                    if isinstance(_crit.rename_variable_column, str)
            )
        if isinstance(drop_levels, str):
            drop_levels = (drop_levels,)
        drop_levels = tuple(set(drop_levels))
        if not isinstance(column_name_source, SummaryColumnSource):
            if not isinstance(column_name_source, str):
                raise TypeError(
                    f"`column_name_source` must be either a `str` or "
                    f"`SummaryColumnSource` enum, not {type(column_name_source)}"
                )
        full_output: dict[str, pd.DataFrame]
        if prepared_output is None:
            if data is None:
                raise ValueError(
                    'If `prepared_output` is `None`, `data` must not be `None`.'
                )
            full_output = self.prepare_output(
                data,
                criteria=criteria,
                columns=columns,
                column_titles=column_titles,
            )
        else:
            if data is not None:
                raise ValueError(
                    'If `prepared_output` is not `None`, `data` must be `None`.'
                )
            full_output = prepared_output
        if len(drop_levels) > 0:
            full_output = {
                _key: _df.droplevel([_level for _level in drop_levels
                                     if _level in _df.index.names])
                    for _key, _df in full_output.items()
            }
        levels_first_df: tuple[str, ...] = tuple()
        for _key, _df in full_output.items():
            if len(levels_first_df) == 0:
                levels_first_df = tuple(_df.index.names)
            if not set(_df.index.names) == set(levels_first_df):
                raise ValueError(
                    'Expected all DataFrames returned by `prepare_output` to '
                    'have the same index levels. The first DataFrame had '
                    f'{levels_first_df}, but found {_df.index.names} for the '
                    f'criterion given by key "{_key}".'
                )
        # Create a helper function to transform the output from a single
        # criterion. Should be a Series, where the correct column has already
        # been selected. The resulting DataFrames will be concatenated
        # horizontally
        def _transform_crit_output(
                _crit_output: pd.Series,
                _column_name_source: SummaryColumnSource|str,
                _dict_key: str,
                _name: str,
        ) -> pd.DataFrame:
            match _column_name_source:
                case SummaryColumnSource.DICT_KEYS:
                    return _crit_output.to_frame(name=_dict_key)
                case SummaryColumnSource.CRITERIA_NAMES:
                    return _crit_output.to_frame(name=_name)
                case levelname if isinstance(levelname, str):
                    return _crit_output.unstack(level=levelname)
                case _:
                    raise TypeError(
                        f'`column_name_source` must be either a `str` or a '
                        '`SummaryColumnSource` enum, not '
                        f'{type(column_name_source)}.'
                    )

        return_dict: dict[str, pd.DataFrame] = {
            summary_keys[_column]: tp.cast(pd.DataFrame, pd.concat(
                [
                    _transform_crit_output(
                        _crit_output=_df[column_titles[_key][_column]],
                        _column_name_source=column_name_source,
                        _dict_key=_key,
                        _name=criteria[_key].name,
                    ) for _key, _df in full_output.items()
                ],
                axis=1,
            )) for _column in columns
        }
        return return_dict

    ###END def MultiCriterionTargetRangeOutput.prepare_summary_output

    def style_output(
            self,
            output: dict[str, pd.DataFrame],
            criteria: tp.Optional[Mapping[str, CriterionTargetRangeTypeVar]] \
                = None,
            column_titles: tp.Optional[Mapping[str, Mapping[CTCol, str]]] \
                = None,
            include_summary: tp.Optional[bool] = None,
            summary_keys: tp.Optional[Mapping[CTCol, str]] = None,
            summary_column_name_source: tp.Optional[SummaryColumnSource] = None,
    ) -> dict[str, PandasStyler]:
        """Style the output from `prepare_output`.

        Parameters
        ----------
        output : dict[str, pandas.DataFrame]
            The output from `prepare_output`.
        criteria : Mapping[str, CriterionTargetRangeTypeVar], optional
            A mapping from  the keys of `output` to `CriterionTargetRange`
            instances. Optional, defaults to `self.criteria`. Must be the same
            criteria instances that were used to prepare the output.
        column_titles : Mapping[str, Mapping[CTCol, str]], optional
            A mapping from the keys of `output` to a mapping from column IDs
            (`CTCol` enums) to the column titles used in the output. Optional,
            defaults to `self._default_column_titles`. Must be the same as the
            column titles used to prepare the output.
        include_summary : bool, optional
            Set to True if `output` includes summary output items, and they
            should be styled with the same styles as the corresponding columns
            in the single-criterion items.
        summary_keys : Mapping[CTCol, str], optional
            A mapping from the column IDs (`CTCol` enums) to the keys of
            `output` that contain the summary columns. Optional, defaults to
            `self._default_summary_keys`. Must be the same as the summary keys
            used to prepare the output.
        summary_column_name_source : SummaryColumnSource, optional
            Specifies where to get the column names of summary DataFrames from.
            Optional, defaults to `self._default_summary_column_name_source`.
            Must be the same as the column name source used to prepare the
            output.
        """
        if criteria is None:
            criteria = self.criteria
        column_titles = self._get_column_titles(
            column_titles=column_titles,
            criteria=criteria,
        )
        reverse_column_titles: dict[str, dict[str, CTCol]] = {
            _key: {v: k for k, v in _ct.items()}
            for _key, _ct in column_titles.items()
        }
        if include_summary is None:
            include_summary = self._default_include_summary
        if summary_keys is None:
            summary_keys = self._default_summary_keys
        if include_summary and not \
                set(summary_keys.values()).isdisjoint(set(criteria.keys())):
            raise ValueError(
                '`summary_keys` has values that overlap with the keys of '
                '`criteria`. This should not be the case, please investigate.'
            )
        if summary_column_name_source is None:
            summary_column_name_source = self._default_summary_column_name_source
        criteria_output: dict[str, CriterionTargetRangeOutputTypeVar] = {
            _key: self.criteria_output_class(
                criteria=_criterion,
                writer=self.writer,
                columns=[
                    reverse_column_titles[_key][_col]
                    for _col in output[_key].columns
                ],
                column_titles=column_titles[_key],
                style=self.style[_key],
            ) for _key, _criterion in criteria.items()
        }
        return_dict: dict[str, PandasStyler] = {
            _key: _crit_output.style_output(output[_key])
            for _key, _crit_output in criteria_output.items()
        }
        if include_summary is True:
            col_crit_output_dict: dict[str, CriterionTargetRangeOutputTypeVar]
            if summary_column_name_source == SummaryColumnSource.DICT_KEYS:
                col_crit_output_dict = {
                    _key: _crit_output
                    for _key, _crit_output in criteria_output.items()
                }
            elif summary_column_name_source == SummaryColumnSource.CRITERIA_NAMES:
                col_crit_output_dict = {
                    _crit_output.criteria.name: _crit_output
                    for _key, _crit_output in criteria_output.items()
                }
            elif isinstance(summary_column_name_source, str):
                raise ValueError(
                    'Styling summary output is not supported when '
                    '`summary_column_name_source` is a string (i.e., and index '
                    'level name).'
                )
            else:
                raise TypeError(
                    f'`summary_column_name_source` must be a '
                    f'`SummaryColumnSource` enum, not '
                    f'{type(summary_column_name_source)}.'
                )
            summary_return_dict: dict[str, PandasStyler] = dict()
            for _ctcol, _output_key in summary_keys.items():
                _output_df: pd.DataFrame = output[_output_key]
                _output_df_style: PandasStyler = _output_df.style
                for _col in _output_df.columns:
                    _output_df_style = \
                        col_crit_output_dict[_col].styling_funcs[_ctcol](
                            _output_df_style, subset=[_col]
                        )
                _output_df_style = self.apply_common_styling(
                    _output_df_style,
                    style=self.style[_output_key],
                )
                summary_return_dict[_output_key] = _output_df_style
            return_dict = summary_return_dict | return_dict
        not_included_items: dict[str, PandasStyler] = {
            _key: _df.style for _key, _df in output.items()
            if _key not in return_dict
        }
        if len(not_included_items) > 0:
            warnings.warn(
                'The items in the `output` parameter with the following keys '
                'were not styled, either because no criterion with the '
                'corresponding key was found in the `criteria` parameter, or '
                'because they were not detected as being summary items to be '
                f'styled: {list(not_included_items.keys())}.'
            )
        return return_dict | not_included_items
    ###END def MultiCriterionTargetRangeOutput.style_output

    def apply_common_styling(
            self,
            styler: PandasStyler,
            *,
            style: tp.Optional[CriterionTargetRangeOutputStyles] = None,
    ) -> PandasStyler:
        """Apply common (not column-specific) styling.

        Applies common styles, such as common number formatting or styling of
        indexes and column headers.

        Note that this method in the current implementation is applied to
        individual DataFrames rather than the whole dictionary of output
        DataFrames, and is only applied to summary DataFrames. If `style_output`
        is called with `include_summary=False`, this method will not be called.'

        Also, this method currently just calls
        `CriterionTargetRangeOutput.apply_common_styling` on the summary
        DataFrames as a shortcut.

        Parameters
        ----------
        styler : pandas Styler
            The pansas Styler to be styled.
        style : CriterionTargetRangeOutputStyles, optional
            The object with styling information to apply.

        Returns
        -------
        styler : pandas Styler
            `styler` with styling applied
        """
        styler = CriterionTargetRangeOutput.apply_common_styling(
            self,
            styler=styler,
            style=style
        )
        return styler
    ###END def MultiCriterionTargetRangeOutput.apply_common_styling

###END class MultiCriterionTargetRangeOutput
